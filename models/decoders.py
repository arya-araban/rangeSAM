# Copyright 2023 - Valeo Comfort and Driving Assistance
# Modifications copyright 2024 - Arya Araban
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .two_way_transformer import TwoWayTransformer


class UpConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, scale_factor=(2, 8), drop_out=False, skip_filters=128):
        super(UpConvBlock, self).__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.skip_filters = skip_filters

        if isinstance(scale_factor, int):
            scale_factor = (scale_factor, scale_factor)
        self.scale_factor = scale_factor

        self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1))

        if drop_out:
            self.dropout = nn.Dropout2d(p=dropout_rate)
        else:
            self.dropout = nn.Identity()

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_filters + skip_filters, out_filters, (3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_filters)
        )

        self.conv_output = nn.Sequential(
            nn.Conv2d(out_filters, out_filters, kernel_size=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_filters),
        )
        if drop_out:
            self.conv_output.add_module('dropout', nn.Dropout2d(p=dropout_rate))

    def forward(self, x, skip=None):
        x = self.conv(x)
        x = self.dropout(x)

        # Upsample to the resolution of the skip connection (the full crop resolution)
        if skip is not None:
            x_up = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        else:
            x_up = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        if self.skip_filters > 0 and skip is not None:
            x_up = torch.cat([x_up, skip], dim=1)

        x_up = self.conv1(x_up)
        return self.conv_output(x_up)

class DecoderUpConv(nn.Module):
    """RangeViT up-convolution decoder followed by a SAM-style two-way transformer that
    lets the point prompts and the image features attend to each other."""

    def __init__(self, n_cls, patch_size, d_encoder, d_decoder, scale_factor=(2, 8), patch_stride=None,
                 dropout_rate=0.2, drop_out=False, skip_filters=128, prompt_embed_dim=128,
                 transformer_depth=3, num_heads=8, mlp_dim=2048, learnable_prompt_scale=True):
        super().__init__()

        self.input_proj = nn.Linear(d_encoder, d_decoder)
        self.up_conv_block = UpConvBlock(8, d_decoder, dropout_rate, scale_factor, drop_out, skip_filters)

        self.two_way_transformer = TwoWayTransformer(
            depth=transformer_depth,
            embedding_dim=d_decoder,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            learnable_prompt_scale=learnable_prompt_scale,
        )
        self.point_proj = nn.Linear(prompt_embed_dim, d_decoder)
        self.image_pe_proj = nn.Linear(d_encoder, d_decoder)
        self.point_norm = nn.LayerNorm(d_decoder)

        self.head = nn.Conv2d(d_decoder, n_cls, kernel_size=(1, 1))

    def forward(self, x, im_size, skip, point_embeddings, image_pe):
        B, L, C = x.shape
        H, W = im_size

        # Project the ViT tokens to the decoder dimension and upsample to full resolution
        x = self.input_proj(x)
        x = x.transpose(1, 2).contiguous().view(B, -1, H, W)
        x = self.up_conv_block(x, skip)

        if point_embeddings is not None:
            x_tokens = x.flatten(2).transpose(1, 2)

            point_embeddings = self.point_norm(self.point_proj(point_embeddings))

            # Positional encodings of the image tokens, resized to the number of pixels
            image_pe = self.image_pe_proj(image_pe).expand(B, -1, -1)
            if image_pe.shape[1] != x_tokens.shape[1]:
                image_pe = F.interpolate(image_pe.transpose(1, 2).unsqueeze(2),
                                         size=(1, x_tokens.shape[1]),
                                         mode='nearest').squeeze(2).transpose(1, 2)

            point_embeddings, x_tokens = self.two_way_transformer(x_tokens, image_pe, point_embeddings)
            x = x_tokens.transpose(1, 2).view(B, -1, H, W)

        return self.head(x)
