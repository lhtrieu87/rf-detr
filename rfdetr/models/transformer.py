# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""
Transformer class
"""

import math
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from rfdetr.models.ops.modules import MSDeformAttn


class MLP(nn.Module):
    """Multi-layer perceptron (MLP) / Feed-Forward Network (FFN)

    A simple feed-forward neural network with ReLU activation between layers.
    Used throughout the transformer for:
    - Reference point head for position encoding
    - Feed-forward layers within transformer blocks
    - Output projection heads

    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output feature dimension
        num_layers (int): Number of linear layers
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # Create list of hidden dimensions: [hidden_dim, hidden_dim, ..., output_dim]
        h = [hidden_dim] * (num_layers - 1)
        # Build sequential linear layers with proper input/output dimensions
        # Zip creates pairs: (input_dim, hidden_dim), (hidden_dim, hidden_dim), ..., (hidden_dim, output_dim)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        """Forward pass through MLP layers

        Applies ReLU activation to all layers except the last one.
        This is standard practice to allow the final layer to output
        any real values (e.g., for regression tasks like bounding box coordinates).
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor, dim=128):
    """Generate sinusoidal position embeddings for 2D or 4D coordinates

    This function creates position embeddings using sine and cosine functions,
    similar to the positional encoding in the original Transformer paper,
    but adapted for 2D spatial positions and optionally width/height dimensions.

    The sinusoidal encoding helps the model understand spatial relationships
    and provides a smooth, continuous representation of positions.

    Args:
        pos_tensor (torch.Tensor): Position coordinates tensor
            - For 2D: shape (..., 2) with [x, y] coordinates
            - For 4D: shape (..., 4) with [x, y, w, h] coordinates
            Values should typically be normalized to [0, 1] range
        dim (int): Dimension of the output encoding for each coordinate (default: 128)
            Final output will have dimension 2*dim for 2D input or 4*dim for 4D input

    Returns:
        torch.Tensor: Sinusoidal position embeddings
            - For 2D input: shape (..., 2*dim)
            - For 4D input: shape (..., 4*dim)

    The encoding uses the formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/dim))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))
    where i ranges from 0 to dim/2
    """
    # Scale coordinates to [0, 2π] range for better sine/cosine coverage
    scale = 2 * math.pi

    # Create frequency bands: [1, 10000^(2/dim), 10000^(4/dim), ..., 10000^(2*(dim//2)/dim)]
    # This creates different frequency scales for encoding position information
    dim_t = torch.arange(dim, dtype=pos_tensor.dtype, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)

    # Extract and scale x, y coordinates
    x_embed = pos_tensor[:, :, 0] * scale  # x coordinates scaled to [0, 2π]
    y_embed = pos_tensor[:, :, 1] * scale  # y coordinates scaled to [0, 2π]

    # Apply frequency encoding: divide positions by frequency bands
    pos_x = x_embed[:, :, None] / dim_t  # Broadcasting: (..., 1) / (dim,) -> (..., dim)
    pos_y = y_embed[:, :, None] / dim_t

    # Apply sine to even indices, cosine to odd indices, then flatten
    # Stack creates (..., dim//2, 2), flatten(2) reshapes to (..., dim)
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)

    if pos_tensor.size(-1) == 2:
        # For 2D coordinates [x, y]: concatenate y and x encodings
        # Note: y comes first, which is a convention in computer vision (height, width)
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        # For 4D coordinates [x, y, w, h]: also encode width and height
        w_embed = pos_tensor[:, :, 2] * scale  # width scaled to [0, 2π]
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale  # height scaled to [0, 2π]
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        # Concatenate all four coordinate encodings: [y, x, w, h]
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def gen_encoder_output_proposals(
    memory, memory_padding_mask, spatial_shapes, unsigmoid=True
):
    """Generate object proposals from encoder features for two-stage detection

    This function creates initial object proposals by generating dense bounding boxes
    over the spatial feature maps. It's used in two-stage DETR variants where the
    encoder produces object proposals that are refined by the decoder.

    The function creates a grid of potential object locations across all feature
    pyramid levels, with box sizes that scale with the pyramid level. This provides
    multi-scale object proposals covering different object sizes.

    Args:
        memory (torch.Tensor): Flattened encoder features from all pyramid levels
            Shape: (batch_size, sum(H_i * W_i), d_model) where sum is over all levels
        memory_padding_mask (torch.Tensor): Mask for padded regions
            Shape: (batch_size, sum(H_i * W_i)) - True for padded/invalid regions
        spatial_shapes (torch.Tensor): Height and width of each feature level
            Shape: (num_levels, 2) with [height, width] for each level
        unsigmoid (bool): Whether to apply inverse sigmoid (logit) transformation
            If True: outputs logits for training, If False: outputs probabilities [0,1]

    Returns:
        tuple: (output_memory, output_proposals)
            - output_memory: Processed encoder features, same shape as input memory
            - output_proposals: Generated box proposals
              Shape: (batch_size, sum(H_i * W_i), 4) in format [cx, cy, w, h]

    The proposal generation works as follows:
    1. For each feature level, create a spatial grid of (x,y) coordinates
    2. Normalize coordinates to [0,1] range based on valid (non-padded) regions
    3. Set box width/height proportional to feature level (larger boxes for coarser levels)
    4. Apply masking to ignore padded regions and invalid proposals
    5. Optionally convert to logit space for training stability
    """
    N_, S_, C_ = memory.shape  # batch_size, total_spatial_locations, channels
    proposals = []
    _cur = 0  # Current index in flattened spatial dimension

    # Process each feature pyramid level
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        # Calculate valid (non-padded) regions for this level
        if memory_padding_mask is not None:
            # Extract mask for current level and reshape to spatial dimensions
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(
                N_, H_, W_, 1
            )
            # Count valid pixels along each dimension
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)  # Valid height per batch
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)  # Valid width per batch
        else:
            # If no padding mask, all regions are valid
            valid_H = torch.tensor([H_ for _ in range(N_)], device=memory.device)
            valid_W = torch.tensor([W_ for _ in range(N_)], device=memory.device)

        # Create spatial grid coordinates for this feature level
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
        )
        grid = torch.cat(
            [grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1
        )  # Shape: (H, W, 2)

        # Normalize grid coordinates to [0,1] based on valid image regions
        # This accounts for different image sizes in the batch after padding
        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(
            N_, 1, 1, 2
        )
        grid = (
            grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5
        ) / scale  # Add 0.5 for center sampling

        # Set proposal box sizes: scale with pyramid level for multi-scale detection
        # Higher levels (coarser features) get larger boxes to detect larger objects
        # 0.05 * 2^level provides exponentially increasing box sizes
        wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)

        # Combine center coordinates and box dimensions: [cx, cy, w, h]
        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += H_ * W_

    # Concatenate proposals from all feature levels
    output_proposals = torch.cat(
        proposals, 1
    )  # Shape: (batch_size, total_locations, 4)

    # Filter valid proposals: boxes should be within reasonable bounds [0.01, 0.99]
    # This removes proposals that are too close to image boundaries
    output_proposals_valid = (
        (output_proposals > 0.01) & (output_proposals < 0.99)
    ).all(-1, keepdim=True)

    if unsigmoid:
        # Convert to logit space: logit(p) = log(p / (1-p))
        # This is more stable for training and matches how box coordinates are typically handled
        output_proposals = torch.log(output_proposals / (1 - output_proposals))

        # Mask invalid regions with positive infinity (will be ignored in loss computation)
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(
                memory_padding_mask.unsqueeze(-1), float("inf")
            )
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float("inf")
        )
    else:
        # Keep in probability space [0,1], mask invalid regions with 0
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(
                memory_padding_mask.unsqueeze(-1), float(0)
            )
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float(0)
        )

    # Process encoder memory features
    output_memory = memory
    if memory_padding_mask is not None:
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0)
        )
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory.to(memory.dtype), output_proposals.to(memory.dtype)


class Transformer(nn.Module):
    """Main Transformer module for RF-DETR object detection

    This is the core transformer architecture that processes multi-scale features from
    the backbone (e.g., DINOv2) and generates object detections. The transformer follows
    the DETR paradigm with several key improvements:

    1. **Multi-scale Deformable Attention**: Efficiently processes features at multiple
       scales using deformable attention mechanisms
    2. **Two-stage Detection**: Optional encoder-decoder architecture where:
       - Stage 1 (Encoder): Generates object proposals from multi-scale features
       - Stage 2 (Decoder): Refines proposals into final detections
    3. **Group DETR Training**: Acceleration technique that groups queries for efficient training
    4. **Reference Point Refinement**: Iteratively improves bounding box predictions

    Key Components:
    - **Encoder**: Processes backbone features (disabled in current implementation)
    - **Decoder**: Refines object queries using cross-attention to features
    - **Two-stage heads**: Optional proposal generation for improved training

    Args:
        d_model (int): Hidden dimension size (default: 512)
        sa_nhead (int): Number of heads for self-attention (default: 8)
        ca_nhead (int): Number of heads for cross-attention (default: 8)
        num_queries (int): Number of object queries (default: 300)
        num_decoder_layers (int): Number of decoder layers (default: 6)
        dim_feedforward (int): Feedforward network dimension (default: 2048)
        dropout (float): Dropout rate (default: 0.0)
        activation (str): Activation function type (default: "relu")
        normalize_before (bool): Whether to normalize before attention (default: False)
        return_intermediate_dec (bool): Return all decoder layer outputs (default: False)
        group_detr (int): Number of query groups for training acceleration (default: 1)
        two_stage (bool): Enable two-stage detection (default: False)
        num_feature_levels (int): Number of feature pyramid levels (default: 4)
        dec_n_points (int): Number of sampling points for deformable attention (default: 4)
        lite_refpoint_refine (bool): Use lightweight reference point refinement (default: False)
        decoder_norm_type (str): Type of normalization in decoder ('LN' or 'Identity')
        bbox_reparam (bool): Use bbox reparameterization for improved training (default: False)
    """

    def __init__(
        self,
        d_model=512,
        sa_nhead=8,
        ca_nhead=8,
        num_queries=300,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        group_detr=1,
        two_stage=False,
        num_feature_levels=4,
        dec_n_points=4,
        lite_refpoint_refine=False,
        decoder_norm_type="LN",
        bbox_reparam=False,
    ):
        super().__init__()

        # Encoder is disabled in current RF-DETR implementation
        # Features are processed by backbone (DINOv2) directly
        self.encoder = None

        # Build decoder layer with multi-scale deformable attention
        decoder_layer = TransformerDecoderLayer(
            d_model,
            sa_nhead,
            ca_nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            group_detr=group_detr,
            num_feature_levels=num_feature_levels,
            dec_n_points=dec_n_points,
            skip_self_attn=False,
        )

        # Configure normalization type for decoder
        assert decoder_norm_type in ["LN", "Identity"]
        norm = {
            "LN": lambda channels: nn.LayerNorm(
                channels
            ),  # Standard layer normalization
            "Identity": lambda channels: nn.Identity(),  # No normalization
        }
        decoder_norm = norm[decoder_norm_type](d_model)

        # Build decoder with multiple layers
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            lite_refpoint_refine=lite_refpoint_refine,
            bbox_reparam=bbox_reparam,
        )

        # Two-stage detection components
        self.two_stage = two_stage
        if two_stage:
            # Separate projection heads for each query group in group DETR
            # These project encoder features to generate initial proposals
            self.enc_output = nn.ModuleList(
                [nn.Linear(d_model, d_model) for _ in range(group_detr)]
            )
            self.enc_output_norm = nn.ModuleList(
                [nn.LayerNorm(d_model) for _ in range(group_detr)]
            )

        # Initialize all parameters with Xavier uniform distribution
        self._reset_parameters()

        # Store configuration for reference
        self.num_queries = num_queries
        self.d_model = d_model
        self.dec_layers = num_decoder_layers
        self.group_detr = group_detr
        self.num_feature_levels = num_feature_levels
        self.bbox_reparam = bbox_reparam

        # Export flag for inference optimization
        self._export = False

    def export(self):
        """Enable export mode for inference optimization

        Sets the transformer to export mode, which modifies the forward pass
        to be more efficient for deployment (e.g., ONNX export).
        """
        self._export = True

    def _reset_parameters(self):
        """Initialize transformer parameters using Xavier uniform distribution

        Applies Xavier uniform initialization to all parameters with more than 1 dimension.
        Also calls specialized initialization for multi-scale deformable attention modules.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def get_valid_ratio(self, mask):
        """Calculate the ratio of valid (non-padded) pixels in feature maps

        This is used to handle variable image sizes in a batch. When images are
        padded to the same size, this function calculates what fraction of each
        dimension contains actual image content vs padding.

        Args:
            mask (torch.Tensor): Boolean mask where True indicates padded regions
                Shape: (batch_size, height, width)

        Returns:
            torch.Tensor: Valid ratios for width and height
                Shape: (batch_size, 2) with [valid_width_ratio, valid_height_ratio]
        """
        _, H, W = mask.shape
        # Count valid (non-masked) pixels along each dimension
        valid_H = torch.sum(~mask[:, :, 0], 1)  # Valid height per sample
        valid_W = torch.sum(~mask[:, 0, :], 1)  # Valid width per sample

        # Convert to ratios [0, 1]
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, refpoint_embed, query_feat):
        """Forward pass of the transformer

        This is the main forward function that processes multi-scale features through
        the transformer decoder to generate object detections.

        Args:
            srcs (List[torch.Tensor]): Multi-scale feature maps from backbone
                Each tensor shape: (batch_size, d_model, height_i, width_i)
                List contains features from different pyramid levels (e.g., P3, P4, P5)
            masks (List[torch.Tensor], optional): Padding masks for each feature level
                Each tensor shape: (batch_size, height_i, width_i)
                True indicates padded/invalid regions
            pos_embeds (List[torch.Tensor]): Position embeddings for each feature level
                Each tensor shape: (batch_size, d_model, height_i, width_i)
                Provides spatial position information to the transformer
            refpoint_embed (torch.Tensor): Reference point embeddings for object queries
                Shape: (num_queries, 4) in normalized coordinates [0,1]
                Represents initial bounding box proposals for each query
            query_feat (torch.Tensor): Initial object query features
                Shape: (num_queries, d_model)
                Learnable embeddings that represent different object "slots"

        Returns:
            tuple: (hs, references, memory_ts, boxes_ts)
                - hs: Decoder outputs for each layer (if return_intermediate=True)
                  Shape: (num_decoder_layers, batch_size, num_queries, d_model)
                - references: Refined reference points after each decoder layer
                  Shape: (num_decoder_layers, batch_size, num_queries, 4)
                - memory_ts: Two-stage memory tokens (if two_stage=True)
                  Shape: (batch_size, num_proposals, d_model)
                - boxes_ts: Two-stage box predictions (if two_stage=True)
                  Shape: (batch_size, num_proposals, 4)
        """
        # Step 1: Process and flatten multi-scale features
        src_flatten = []
        mask_flatten = [] if masks is not None else None
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = [] if masks is not None else None

        # Process each feature pyramid level (e.g., P3, P4, P5)
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
            # This converts 2D feature maps to sequences for transformer processing
            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed_flatten.append(pos_embed)
            src_flatten.append(src)

            # Flatten masks if provided
            if masks is not None:
                mask = masks[lvl].flatten(1)  # bs, hw
                mask_flatten.append(mask)

        # Step 2: Concatenate all feature levels into single sequences
        # Concatenate features from all pyramid levels: P3 + P4 + P5 -> single sequence
        memory = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        if masks is not None:
            mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
            # Calculate valid ratios for each pyramid level to handle padding
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c

        # Create spatial metadata for deformable attention
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=memory.device
        )
        # Calculate starting index for each feature level in the flattened sequence
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        # Step 3: Two-stage detection (optional)
        # Generate initial object proposals from encoder features
        if self.two_stage:
            # Generate dense object proposals across all feature locations
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes, unsigmoid=not self.bbox_reparam
            )

            # Group DETR processing: split queries into groups for training efficiency
            # During inference, only use 1 group for efficiency
            refpoint_embed_ts, memory_ts, boxes_ts = [], [], []
            group_detr = self.group_detr if self.training else 1

            for g_idx in range(group_detr):
                # Process encoder features through group-specific projection heads
                # This creates specialized representations for each query group
                output_memory_gidx = self.enc_output_norm[g_idx](
                    self.enc_output[g_idx](output_memory)
                )

                # Generate classification scores for proposal selection
                # enc_out_class_embed and enc_out_bbox_embed are set externally (likely in main model)
                enc_outputs_class_unselected_gidx = self.enc_out_class_embed[g_idx](
                    output_memory_gidx
                )

                # Generate bounding box predictions
                if self.bbox_reparam:
                    # Bbox reparameterization: predict deltas and apply them to proposals
                    # This improves training stability and convergence
                    enc_outputs_coord_delta_gidx = self.enc_out_bbox_embed[g_idx](
                        output_memory_gidx
                    )

                    # Apply delta predictions to proposal coordinates
                    # Delta format: [delta_cx, delta_cy, delta_w, delta_h]
                    enc_outputs_coord_cxcy_gidx = (
                        enc_outputs_coord_delta_gidx[..., :2]
                        * output_proposals[..., 2:]
                        + output_proposals[..., :2]
                    )
                    enc_outputs_coord_wh_gidx = (
                        enc_outputs_coord_delta_gidx[..., 2:].exp()
                        * output_proposals[..., 2:]
                    )
                    enc_outputs_coord_unselected_gidx = torch.concat(
                        [enc_outputs_coord_cxcy_gidx, enc_outputs_coord_wh_gidx], dim=-1
                    )
                else:
                    # Direct coordinate prediction: add deltas to proposals
                    enc_outputs_coord_unselected_gidx = (
                        self.enc_out_bbox_embed[g_idx](output_memory_gidx)
                        + output_proposals
                    )  # (bs, \sum{hw}, 4) unsigmoid

                # Select top-k proposals based on classification confidence
                # This filters the dense proposals to the most promising ones
                topk = min(
                    self.num_queries, enc_outputs_class_unselected_gidx.shape[-2]
                )
                topk_proposals_gidx = torch.topk(
                    enc_outputs_class_unselected_gidx.max(-1)[0], topk, dim=1
                )[1]  # bs, nq

                # Extract selected proposal coordinates and features
                refpoint_embed_gidx_undetach = torch.gather(
                    enc_outputs_coord_unselected_gidx,
                    1,
                    topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, 4),
                )  # unsigmoid
                # Detach for decoder initialization (stop gradients)
                # Total Loss = loss_bbox + loss_giou + loss_labels +
                # loss_bbox_enc + loss_giou_enc + loss_labels_enc + ...
                # This dual-loss strategy allows RF-DETR to leverage
                # both dense spatial coverage (encoder) and
                # iterative refinement (decoder) while maintaining training stability.
                refpoint_embed_gidx = refpoint_embed_gidx_undetach.detach()

                # Extract corresponding memory features for selected proposals
                tgt_undetach_gidx = torch.gather(
                    output_memory_gidx,
                    1,
                    topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, self.d_model),
                )

                # Collect results from this query group
                refpoint_embed_ts.append(refpoint_embed_gidx)
                memory_ts.append(tgt_undetach_gidx)
                boxes_ts.append(refpoint_embed_gidx_undetach)

            # Concatenate results from all query groups
            # This combines the proposals from different groups into final sets
            # total_proposals = group_detr * num_queries
            # This Group DETR technique allows the model to
            # generate more diverse proposals during training
            # while maintaining efficiency during inference.
            refpoint_embed_ts = torch.cat(
                refpoint_embed_ts, dim=1
            )  # (bs, total_proposals, 4)
            memory_ts = torch.cat(memory_ts, dim=1)  # (bs, total_proposals, d_model)
            boxes_ts = torch.cat(boxes_ts, dim=1)  # (bs, total_proposals, 4)

        # Step 4: Decoder processing
        # The decoder refines object queries through cross-attention with features
        if self.dec_layers > 0:
            # Prepare initial decoder inputs
            # Expand query features and reference points for batch processing
            tgt = query_feat.unsqueeze(0).repeat(bs, 1, 1)  # (bs, num_queries, d_model)
            refpoint_embed = refpoint_embed.unsqueeze(0).repeat(
                bs, 1, 1
            )  # (bs, num_queries, 4)

            if self.two_stage:
                # Combine two-stage proposals with learned queries
                # Two-stage provides initial proposals, learned queries add diversity
                ts_len = refpoint_embed_ts.shape[-2]
                refpoint_embed_ts_subset = refpoint_embed[
                    ..., :ts_len, :
                ]  # Two-stage proposals
                refpoint_embed_subset = refpoint_embed[
                    ..., ts_len:, :
                ]  # Learned queries

                if self.bbox_reparam:
                    # Apply bbox reparameterization to two-stage proposals
                    refpoint_embed_cxcy = (
                        refpoint_embed_ts_subset[..., :2] * refpoint_embed_ts[..., 2:]
                    )
                    refpoint_embed_cxcy = (
                        refpoint_embed_cxcy + refpoint_embed_ts[..., :2]
                    )
                    refpoint_embed_wh = (
                        refpoint_embed_ts_subset[..., 2:].exp()
                        * refpoint_embed_ts[..., 2:]
                    )
                    refpoint_embed_ts_subset = torch.concat(
                        [refpoint_embed_cxcy, refpoint_embed_wh], dim=-1
                    )
                else:
                    # Simple addition for coordinate refinement
                    refpoint_embed_ts_subset = (
                        refpoint_embed_ts_subset + refpoint_embed_ts
                    )

                # Combine two-stage and learned query reference points
                refpoint_embed = torch.concat(
                    [refpoint_embed_ts_subset, refpoint_embed_subset], dim=-2
                )

            # Run decoder: iteratively refine queries through cross-attention
            # The decoder attends to multi-scale features to produce final detections
            hs, references = self.decoder(
                tgt,
                memory,
                memory_key_padding_mask=mask_flatten,
                pos=lvl_pos_embed_flatten,
                refpoints_unsigmoid=refpoint_embed,
                level_start_index=level_start_index,
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios.to(memory.dtype)
                if valid_ratios is not None
                else valid_ratios,
            )
        else:
            # No decoder layers: only use two-stage proposals
            assert self.two_stage, "if not using decoder, two_stage must be True"
            hs = None
            references = None

        # Step 5: Return results
        if self.two_stage:
            # Return both decoder outputs and two-stage proposals
            if self.bbox_reparam:
                # Keep boxes in logit space for reparameterized training
                return hs, references, memory_ts, boxes_ts
            else:
                # Convert boxes to [0,1] probability space
                return hs, references, memory_ts, boxes_ts.sigmoid()

        # Return only decoder outputs (single-stage)
        return hs, references, None, None


class TransformerDecoder(nn.Module):
    """Multi-layer transformer decoder for object detection

    This decoder consists of multiple TransformerDecoderLayer modules stacked together.
    Each layer performs self-attention among object queries and cross-attention between
    queries and multi-scale feature maps. The decoder iteratively refines object
    representations and their bounding box coordinates.

    Key Features:
    1. **Iterative Refinement**: Each layer refines the bounding box predictions
    2. **Reference Point Updates**: Box coordinates are progressively improved
    3. **Multi-scale Cross-attention**: Queries attend to features at multiple scales
    4. **Position-aware Processing**: Uses positional embeddings for spatial reasoning

    Args:
        decoder_layer: A single transformer decoder layer to be replicated
        num_layers (int): Number of decoder layers to stack
        norm (nn.Module, optional): Final normalization layer
        return_intermediate (bool): Whether to return outputs from all layers
        d_model (int): Hidden dimension size
        lite_refpoint_refine (bool): Use lightweight reference point refinement
        bbox_reparam (bool): Use bounding box reparameterization
    """

    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        d_model=256,
        lite_refpoint_refine=False,
        bbox_reparam=False,
    ):
        super().__init__()
        # Create multiple identical decoder layers
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.lite_refpoint_refine = lite_refpoint_refine
        self.bbox_reparam = bbox_reparam

        # Reference point head: converts positional encodings to query position embeddings
        # Takes 2*d_model input (sine embeddings) and outputs d_model position features
        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)

        self._export = False

    def export(self):
        """Enable export mode for inference optimization"""
        self._export = True

    def refpoints_refine(self, refpoints_unsigmoid, new_refpoints_delta):
        """Apply reference point refinement using predicted deltas

        This function updates bounding box coordinates by applying predicted
        deltas to the current reference points. The refinement strategy depends
        on whether bbox reparameterization is enabled.

        Args:
            refpoints_unsigmoid (torch.Tensor): Current reference points in logit space
                Shape: (batch_size, num_queries, 4) with [cx, cy, w, h]
            new_refpoints_delta (torch.Tensor): Predicted coordinate deltas
                Shape: (batch_size, num_queries, 4)

        Returns:
            torch.Tensor: Refined reference points in logit space
                Shape: (batch_size, num_queries, 4)
        """
        if self.bbox_reparam:
            # Reparameterized refinement: more stable training
            # Delta[:2] modifies center coordinates: delta_cxcy * old_wh + old_cxcy
            # Delta[2:] modifies width/height: exp(delta_wh) * old_wh
            # Instead of predicting absolute movement, you predict a fraction of box width/height.
            # So the model learns normalized deltas independent of image scale.
            # Centers (cx,cy) get linear updates relative to size → translation.
            new_refpoints_cxcy = (
                new_refpoints_delta[..., :2] * refpoints_unsigmoid[..., 2:]
                + refpoints_unsigmoid[..., :2]
            )

            # Multiplicative change of width/height using exponentials ensures positivity.
            # Predicting in log-space stabilizes training because size ratios vary widely but logs do not.
            # Width/height (w,h) get exponential updates → scale transformation.
            new_refpoints_wh = (
                new_refpoints_delta[..., 2:].exp() * refpoints_unsigmoid[..., 2:]
            )
            new_refpoints_unsigmoid = torch.concat(
                [new_refpoints_cxcy, new_refpoints_wh], dim=-1
            )
        else:
            # Direct delta addition: simpler but potentially less stable
            new_refpoints_unsigmoid = refpoints_unsigmoid + new_refpoints_delta
        return new_refpoints_unsigmoid

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        refpoints_unsigmoid: Optional[Tensor] = None,
        # for memory
        level_start_index: Optional[Tensor] = None,  # num_levels
        spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        valid_ratios: Optional[Tensor] = None,
    ):
        """Forward pass through the multi-layer transformer decoder

        This method processes object queries through multiple decoder layers, where each
        layer performs self-attention among queries and cross-attention with multi-scale
        feature maps. The decoder iteratively refines object representations and updates
        their reference point coordinates.

        Args:
            tgt (torch.Tensor): Initial object query features
                Shape: (batch_size, num_queries, d_model)
            memory (torch.Tensor): Concatenated multi-scale encoder features
                Shape: (batch_size, sum(H_i*W_i), d_model)
            tgt_mask (torch.Tensor, optional): Target attention mask
            memory_mask (torch.Tensor, optional): Memory attention mask
            tgt_key_padding_mask (torch.Tensor, optional): Target padding mask
            memory_key_padding_mask (torch.Tensor, optional): Memory padding mask
                Shape: (batch_size, sum(H_i*W_i)) - True for padded regions
            pos (torch.Tensor, optional): Position embeddings for memory features
                Shape: (batch_size, sum(H_i*W_i), d_model)
            refpoints_unsigmoid (torch.Tensor, optional): Reference points in logit space
                Shape: (batch_size, num_queries, 4) with [cx, cy, w, h] coordinates
            level_start_index (torch.Tensor, optional): Starting indices for each feature level
                Shape: (num_levels,) - indices where each pyramid level starts in memory
            spatial_shapes (torch.Tensor, optional): Spatial dimensions of each level
                Shape: (num_levels, 2) with [height, width] for each pyramid level
            valid_ratios (torch.Tensor, optional): Valid pixel ratios for each level
                Shape: (batch_size, num_levels, 2) with [width_ratio, height_ratio]

        Returns:
            tuple: Decoder outputs and reference points
                If return_intermediate=True:
                    - outputs: (num_layers, batch_size, num_queries, d_model)
                    - references: (num_layers, batch_size, num_queries, 4)
                If return_intermediate=False:
                    - outputs: (1, batch_size, num_queries, d_model)
                    - references: (1, batch_size, num_queries, 4)
        """
        output = tgt

        # Storage for intermediate results if requested
        intermediate = []
        hs_refpoints_unsigmoid = [refpoints_unsigmoid]

        def get_reference(refpoints):
            # Converts “where each object query looks” into “what spatial features to attend to.”
            """Generate position-aware query embeddings from reference points

            This inner function converts bounding box coordinates into positional
            embeddings that guide the attention mechanisms. It handles both
            export mode (inference) and training mode differently for efficiency.

            Args:
                refpoints (torch.Tensor): Reference points
                    Shape: (batch_size, num_queries, 4)

            Returns:
                tuple: (obj_center, refpoints_input, query_pos, query_sine_embed)
                    - obj_center: Normalized box coordinates for loss computation
                    - refpoints_input: Multi-scale reference points for deformable attention
                    - query_pos: Position embeddings for query features
                    - query_sine_embed: Raw sinusoidal position embeddings
            """
            # Extract center coordinates and box dimensions
            obj_center = refpoints[..., :4]

            if self._export:
                # Export mode: simplified processing for inference efficiency
                query_sine_embed = gen_sineembed_for_position(
                    obj_center, self.d_model // 2
                )  # bs, nq, 256*2
                # Feature maps already valid everywhere because you usually process one image at a time.
                # It’s often resized but not padded. Do not need to adjust for valid ratios.
                refpoints_input = obj_center[:, :, None]  # bs, nq, 1, 4
            else:
                # Training mode: account for different image sizes and padding
                # Scale reference points by valid ratios for each pyramid level
                # If you don’t compensate, a normalized x = 0.9 (right edge of original image)
                # would point into the padded region on the feature map, where there are no real features.
                refpoints_input = (
                    obj_center[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )  # bs, nq, nlevel, 4
                # Generate sinusoidal embeddings for the first level only (more efficient)
                # All pyramid levels share the same query position embedding.
                query_sine_embed = gen_sineembed_for_position(
                    refpoints_input[:, :, 0, :], self.d_model // 2
                )  # bs, nq, 256*2

            # Convert sinusoidal embeddings to learnable position features
            query_pos = self.ref_point_head(query_sine_embed)
            return obj_center, refpoints_input, query_pos, query_sine_embed

        # Generate initial position embeddings
        if self.lite_refpoint_refine:
            # Lite mode: compute position embeddings once and reuse across layers
            if self.bbox_reparam:
                obj_center, refpoints_input, query_pos, query_sine_embed = (
                    get_reference(refpoints_unsigmoid)
                )
            else:
                obj_center, refpoints_input, query_pos, query_sine_embed = (
                    get_reference(refpoints_unsigmoid.sigmoid())
                )

        # Process through each decoder layer
        for layer_id, layer in enumerate(self.layers):
            if not self.lite_refpoint_refine:
                # Standard mode: recompute position embeddings for each layer
                # This allows for dynamic position updates but is more expensive
                if self.bbox_reparam:
                    obj_center, refpoints_input, query_pos, query_sine_embed = (
                        get_reference(refpoints_unsigmoid)
                    )
                else:
                    obj_center, refpoints_input, query_pos, query_sine_embed = (
                        get_reference(refpoints_unsigmoid.sigmoid())
                    )

            # Apply positional transformation (currently identity, could be layer-specific)
            pos_transformation = 1
            query_pos = query_pos * pos_transformation

            # Process through decoder layer
            # (batch_size, num_queries, d_model)
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                query_sine_embed=query_sine_embed,
                is_first=(layer_id == 0),
                reference_points=refpoints_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )

            if not self.lite_refpoint_refine:
                # Iterative bounding box refinement
                # Each layer predicts deltas to improve the reference points
                # Initialize in the ancestor lwdetr
                # (batch_size, num_queries, 4)
                new_refpoints_delta = self.bbox_embed(output)
                new_refpoints_unsigmoid = self.refpoints_refine(
                    refpoints_unsigmoid, new_refpoints_delta
                )

                # Store intermediate reference points (except for last layer)
                if layer_id != self.num_layers - 1:
                    hs_refpoints_unsigmoid.append(new_refpoints_unsigmoid)

                # Update reference points for next layer (detach to stop gradients)
                refpoints_unsigmoid = new_refpoints_unsigmoid.detach()

            # Store intermediate decoder outputs if requested
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        # Apply final normalization
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                # Replace last intermediate output with normalized version
                intermediate.pop()
                intermediate.append(output)

        # Return results based on mode and configuration
        if self.return_intermediate:
            if self._export:
                # Export mode: return only final layer results for efficiency
                hs = intermediate[-1]  # Shape: (batch_size, num_queries, d_model)
                if self.bbox_embed is not None:
                    ref = hs_refpoints_unsigmoid[-1]
                else:
                    ref = refpoints_unsigmoid
                return hs, ref

            # Training/inference mode: return all intermediate results
            if self.bbox_embed is not None:
                # Return all decoder layer outputs and refined reference points
                return [
                    torch.stack(intermediate),  # (num_layers, bs, nq, d_model)
                    torch.stack(hs_refpoints_unsigmoid),  # (num_layers, bs, nq, 4)
                ]
            else:
                # Return decoder outputs with initial reference points
                return [
                    torch.stack(intermediate),
                    refpoints_unsigmoid.unsqueeze(0),  # (1, bs, nq, 4)
                ]

        # Return only final layer output
        return output.unsqueeze(0)  # (1, bs, nq, d_model)


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with self-attention and cross-attention

    This layer implements a single decoder block in the transformer architecture.
    Each layer consists of three main components:

    1. **Self-Attention**: Object queries attend to each other to model relationships
    2. **Cross-Attention**: Object queries attend to multi-scale feature maps
    3. **Feed-Forward Network**: Point-wise processing for feature refinement

    The layer uses multi-scale deformable attention for efficient cross-attention
    with feature pyramids, and supports group DETR for training acceleration.

    Args:
        d_model (int): Hidden dimension size
        sa_nhead (int): Number of heads for self-attention
        ca_nhead (int): Number of heads for cross-attention
        dim_feedforward (int): Feed-forward network dimension (default: 2048)
        dropout (float): Dropout rate (default: 0.1)
        activation (str): Activation function type (default: "relu")
        normalize_before (bool): Apply normalization before attention (default: False)
        group_detr (int): Number of query groups for training acceleration (default: 1)
        num_feature_levels (int): Number of feature pyramid levels (default: 4)
        dec_n_points (int): Number of sampling points for deformable attention (default: 4)
        skip_self_attn (bool): Skip self-attention computation (default: False)
    """

    def __init__(
        self,
        d_model,
        sa_nhead,
        ca_nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        group_detr=1,
        num_feature_levels=4,
        dec_n_points=4,
        skip_self_attn=False,
    ):
        super().__init__()

        # Self-Attention Module
        # Standard multi-head attention among object queries
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=sa_nhead, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-Attention Module
        # Multi-scale deformable attention for attending to feature pyramids
        self.cross_attn = MSDeformAttn(
            d_model,
            n_levels=num_feature_levels,
            n_heads=ca_nhead,
            n_points=dec_n_points,
        )

        self.nhead = ca_nhead

        # Feed-Forward Network
        # Two linear layers with activation in between
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers for residual connections
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Additional dropout layers
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Configuration
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.group_detr = group_detr

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """Add positional embeddings to tensor if provided

        This is a utility function to conditionally add positional information.
        Positional embeddings help the model understand spatial relationships.
        """
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        query_sine_embed=None,
        is_first=False,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
    ):
        """Forward pass with post-normalization (standard transformer approach)

        This implements the "post-norm" variant where normalization is applied
        after the attention and feed-forward operations, rather than before.

        Processing Flow:
        1. Self-attention among object queries with residual connection
        2. Cross-attention between queries and multi-scale features
        3. Feed-forward processing with residual connection
        4. Normalization applied after each operation

        Args:
            tgt (torch.Tensor): Object query features
                Shape: (batch_size, num_queries, d_model)
            memory (torch.Tensor): Multi-scale encoder features
                Shape: (batch_size, sum(H_i*W_i), d_model)
            query_pos (torch.Tensor): Position embeddings for queries
                Shape: (batch_size, num_queries, d_model)
            reference_points (torch.Tensor): Reference points for deformable attention
                Shape: (batch_size, num_queries, num_levels, 4)
            spatial_shapes (torch.Tensor): Spatial dimensions of feature levels
            level_start_index (torch.Tensor): Starting indices for each feature level
            memory_key_padding_mask (torch.Tensor): Padding mask for memory features
            Other args: Standard transformer layer arguments

        Returns:
            torch.Tensor: Refined object query features
                Shape: (batch_size, num_queries, d_model)
        """
        bs, num_queries, _ = tgt.shape

        # ========== Self-Attention Block =============
        # Object queries attend to each other to model relationships
        # Apply positional embeddings to queries for spatial awareness
        q = k = tgt + query_pos  # Add position info to queries and keys
        v = tgt  # Values remain as original features

        # Group DETR optimization: during training, split queries into groups
        # This reduces memory usage and computational cost for large query sets
        if self.training:
            # Split queries into groups along the query dimension
            # (bs * self.group_detr, num_queries // self.group_detr, d_model)
            q = torch.cat(q.split(num_queries // self.group_detr, dim=1), dim=0)
            k = torch.cat(k.split(num_queries // self.group_detr, dim=1), dim=0)
            v = torch.cat(v.split(num_queries // self.group_detr, dim=1), dim=0)

        # Standard multi-head self-attention
        tgt2 = self.self_attn(
            q,
            k,
            v,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )[0]

        # Recombine groups if training
        if self.training:
            tgt2 = torch.cat(tgt2.split(bs, dim=0), dim=1)
        # ========== End of Self-Attention =============

        # Apply residual connection and normalization
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ========== Cross-Attention Block =============
        # Object queries attend to multi-scale feature maps
        # This is where queries gather information from the image features
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),  # Add positional info to queries
            reference_points,  # Where to attend in feature maps
            memory,  # Multi-scale feature maps
            spatial_shapes,  # Spatial dimensions of each level
            level_start_index,  # Starting index of each level
            memory_key_padding_mask,  # Mask for padded regions
        )
        # ========== End of Cross-Attention =============

        # Apply residual connection and normalization
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ========== Feed-Forward Block =============
        # Point-wise processing for feature refinement
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        # ========== End of Feed-Forward =============

        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        query_sine_embed=None,
        is_first=False,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
    ):
        """Forward pass through decoder layer (delegates to forward_post)

        This is the main forward method that delegates to the post-normalization
        implementation. Could be extended to support pre-normalization variants.
        """
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
            query_sine_embed,
            is_first,
            reference_points,
            spatial_shapes,
            level_start_index,
        )


def _get_clones(module, N):
    """Create N identical copies of a PyTorch module

    This utility function creates multiple independent copies of a module,
    each with their own parameters. Used to create multiple decoder layers.

    Args:
        module (nn.Module): The module to replicate
        N (int): Number of copies to create

    Returns:
        nn.ModuleList: List containing N independent copies of the module
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    """Factory function to build a complete transformer from configuration

    This function constructs a Transformer instance using parameters from
    a configuration object. It handles optional parameters gracefully and
    provides reasonable defaults.

    Args:
        args: Configuration object with transformer parameters
            Expected attributes include:
            - hidden_dim: Model dimension (d_model)
            - sa_nheads, ca_nheads: Number of attention heads
            - num_queries: Number of object queries
            - dropout: Dropout rate
            - dim_feedforward: FFN dimension
            - dec_layers: Number of decoder layers
            - group_detr: Number of query groups for training
            - two_stage: Enable two-stage detection (optional)
            - num_feature_levels: Number of pyramid levels
            - dec_n_points: Deformable attention sampling points
            - lite_refpoint_refine: Use lightweight refinement
            - decoder_norm: Decoder normalization type
            - bbox_reparam: Enable bbox reparameterization

    Returns:
        Transformer: Configured transformer instance ready for training/inference
    """

    # Handle optional two_stage parameter with fallback
    try:
        two_stage = args.two_stage
    except AttributeError:
        two_stage = False

    return Transformer(
        d_model=args.hidden_dim,
        sa_nhead=args.sa_nheads,
        ca_nhead=args.ca_nheads,
        num_queries=args.num_queries,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
        group_detr=args.group_detr,
        two_stage=two_stage,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        lite_refpoint_refine=args.lite_refpoint_refine,
        decoder_norm_type=args.decoder_norm,
        bbox_reparam=args.bbox_reparam,
    )


def _get_activation_fn(activation):
    """Get activation function by name

    Returns the corresponding PyTorch activation function for a given string name.
    Supports the most commonly used activation functions in transformers.

    Args:
        activation (str): Name of activation function
            Supported values: "relu", "gelu", "glu"

    Returns:
        callable: The activation function

    Raises:
        RuntimeError: If activation name is not supported
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
