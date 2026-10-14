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
import timm
from timm.layers import trunc_normal_

from .prompt_encoder import PromptEncoder
from .blocks import Block
from .model_utils import adapt_input_conv, padding, unpadding, resize_pos_embed, init_weights
from .stems import PatchEmbedding, ConvStem
from .decoders import DecoderUpConv


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        d_ff,
        n_heads,
        n_cls,
        dropout=0.1,
        drop_path_rate=0.0,
        channels=3,
        ls_init_values=None,
        patch_stride=None,
        conv_stem='none',
        stem_base_channels=32,
        stem_hidden_dim=None,
    ):
        super().__init__()

        self.conv_stem = conv_stem

        if self.conv_stem == 'none':
            self.patch_embed = PatchEmbedding(
                image_size,
                patch_size,
                patch_stride,
                d_model,
                channels,)
        else:   # in this case self.conv_stem = 'ConvStem'
            assert patch_stride == patch_size # patch_size = patch_stride if a convolutional stem is used

            self.patch_embed = ConvStem(
                in_channels=channels,
                base_channels=stem_base_channels,
                img_size=image_size,
                patch_stride=patch_stride,
                embed_dim=d_model,
                flatten=True,
                hidden_dim=stem_hidden_dim)

        self.patch_size = patch_size
        self.PS_H, self.PS_W = patch_size
        self.patch_stride = patch_stride
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls
        self.image_size = image_size

        # cls and pos tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.num_patches + 1, d_model))

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
                [Block(d_model, n_heads, d_ff, dropout, dpr[i], init_values=ls_init_values) for i in range(n_layers)]
            )

        self.norm = nn.LayerNorm(d_model)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_grid_size(self, H, W):
        return self.patch_embed.get_grid_size(H, W)

    def forward(self, im, return_features=False):
        B, _, H, W = im.shape
        x, skip = self.patch_embed(im) # x.shape = [16, 576, 384]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # x.shape = [16, 577, 384]

        pos_embed = self.pos_embed
        num_extra_tokens = 1

        if x.shape[1] != pos_embed.shape[1]:
            grid_H, grid_W = self.get_grid_size(H, W)
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (grid_H, grid_W),
                num_extra_tokens,
            )

        x = x + pos_embed
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x, skip  # x.shape = [16, 577, 384]


def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    model_cfg.pop('backbone')
    mlp_expansion_ratio = 4
    model_cfg['d_ff'] = mlp_expansion_ratio * model_cfg['d_model']

    new_patch_size = model_cfg.pop('new_patch_size')
    new_patch_stride = model_cfg.pop('new_patch_stride')

    if (new_patch_size is not None):
        if new_patch_stride is None:
            new_patch_stride = new_patch_size
        model_cfg['patch_size'] = new_patch_size
        model_cfg['patch_stride'] = new_patch_stride

    model = VisionTransformer(**model_cfg)

    return model


def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop('name')
    if name != 'up_conv':
        raise ValueError(f'Unknown decoder: {name}')
    decoder_cfg['d_encoder'] = encoder.d_model
    decoder_cfg['patch_size'] = encoder.patch_size
    decoder_cfg['patch_stride'] = encoder.patch_stride
    return DecoderUpConv(**decoder_cfg)


def create_rangevit(model_cfg):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop('decoder')
    decoder_cfg['n_cls'] = model_cfg['n_cls']

    encoder = create_vit(model_cfg)
    decoder = create_decoder(encoder, decoder_cfg)
    return RangeViTBackbone(encoder, decoder, n_cls=model_cfg['n_cls'])


class RangeViTBackbone(nn.Module):
    """Convolutional stem + ViT encoder and the prompt-aware decoder."""

    def __init__(self, encoder, decoder, n_cls):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.patch_stride = encoder.patch_stride
        self.encoder = encoder
        self.decoder = decoder

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay('encoder.', self.encoder).union(
            append_prefix_no_weight_decay('decoder.', self.decoder)
        )
        return nwd_params


class RangeSAM(nn.Module):
    """RangeSAM: promptable, class-agnostic segmentation of LiDAR range images.

    Built on RangeViT (ViT encoder with a convolutional stem) with a SAM-style point
    prompt encoder and a two-way transformer in the decoder. The three components
    introduced by RangeSAM can be switched off individually for ablations:
      - label_enforcement: prompted pixels are forced to their prompt label
      - learnable_prompt_scale: learnable scaling of prompt/image attention
      - the prompt loss (see train.py, `use_prompt_loss`)
    """

    def __init__(
        self,
        in_channels=5,
        n_cls=17,
        backbone='vit_small_patch16_384',
        image_size=(32, 384),
        pretrained_path=None,
        new_patch_size=None,
        new_patch_stride=None,
        reuse_pos_emb=False,
        reuse_patch_emb=False,
        conv_stem='none',
        stem_base_channels=32,
        stem_hidden_dim=None,
        skip_filters=0,
        decoder='up_conv',
        up_conv_d_decoder=64,
        up_conv_scale_factor=(2, 8),
        prompt_embed_dim=128,
        transformer_depth=3,
        num_heads=8,
        mlp_dim=2048,
        label_enforcement=True,
        learnable_prompt_scale=True,
        ):
        super().__init__()

        self.n_cls = n_cls
        self.label_enforcement = label_enforcement

        if backbone == 'vit_small_patch16_384':
            n_heads = 6
            n_layers = 12
            patch_size = 16
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 384
        elif backbone == 'vit_base_patch16_384':
            n_heads = 12
            n_layers = 12
            patch_size = 16
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 768
        elif backbone == 'vit_large_patch16_384':
            n_heads = 16
            n_layers = 24
            patch_size = 16
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 1024
        else:
            raise NameError('Not known ViT backbone.')

        # Decoder config
        decoder_cfg = {
            'n_cls': n_cls, 'name': decoder,
            'd_decoder': up_conv_d_decoder, # hidden dim of the decoder
            'scale_factor': up_conv_scale_factor, # upsampling factor of the decoder
            'skip_filters': skip_filters,} # channel dim of the skip connection (between the convolutional stem and the decoder)

        self.prompt_encoder = PromptEncoder(prompt_embed_dim, image_size)

        # Prompt / two-way transformer settings of the decoder
        decoder_cfg['prompt_embed_dim'] = prompt_embed_dim
        decoder_cfg['transformer_depth'] = transformer_depth
        decoder_cfg['num_heads'] = num_heads
        decoder_cfg['mlp_dim'] = mlp_dim
        decoder_cfg['learnable_prompt_scale'] = learnable_prompt_scale

        # ViT encoder and stem config
        net_kwargs = {
            'backbone': backbone,
            'd_model': d_model, # dim of features
            'decoder': decoder_cfg,
            'drop_path_rate': drop_path_rate,
            'dropout': dropout,
            'channels': in_channels, # nb of channels for the 3D point projections
            'image_size': image_size,
            'n_cls': n_cls,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'patch_size': patch_size, # old patch size for the ViT encoder
            'new_patch_size': new_patch_size, # new patch size for the ViT encoder
            'new_patch_stride': new_patch_stride, # new patch stride for the ViT encoder
            'conv_stem': conv_stem,
            'stem_base_channels': stem_base_channels,
            'stem_hidden_dim': stem_hidden_dim,
        }

        # Create the RangeViT backbone (encoder + decoder)
        self.rangevit = create_rangevit(net_kwargs)

        old_state_dict = self.rangevit.state_dict()

        # Loading pre-trained weights in the ViT encoder
        if pretrained_path is not None:
            print(f'Loading pretrained parameters from {pretrained_path}')
            if pretrained_path == 'timmImageNet21k':
                vit_imagenet = timm.create_model(backbone, pretrained=True) #.cuda()
                pretrained_state_dict = vit_imagenet.state_dict() # nb keys: 152
                all_keys = list(pretrained_state_dict.keys())
                for key in all_keys:
                    pretrained_state_dict['encoder.'+key] = pretrained_state_dict.pop(key)
            else:
                pretrained_state_dict = torch.load(pretrained_path, map_location='cpu')
                if 'model' in pretrained_state_dict:
                    pretrained_state_dict = pretrained_state_dict['model']
                elif 'pos_embed' in pretrained_state_dict.keys():
                    all_keys = list(pretrained_state_dict.keys())
                    for key in all_keys:
                        pretrained_state_dict['encoder.'+key] = pretrained_state_dict.pop(key)

            # Reuse pre-trained positional embeddings
            if reuse_pos_emb:
                # Resize the existing position embeddings to the desired size
                print('Reusing positional embeddings.')
                gs_new_h = int((image_size[0] - new_patch_size[0]) // new_patch_stride[0] + 1)
                gs_new_w = int((image_size[1] - new_patch_size[1]) // new_patch_stride[1] + 1)
                num_extra_tokens = 1
                resized_pos_emb = resize_pos_embed(pretrained_state_dict['encoder.pos_embed'],
                                                   grid_old_shape=None,
                                                   grid_new_shape=(gs_new_h, gs_new_w),
                                                   num_extra_tokens=num_extra_tokens)
                pretrained_state_dict['encoder.pos_embed'] = resized_pos_emb
            else:
                del pretrained_state_dict['encoder.pos_embed'] # remove positional embeddings

            # Reuse pre-trained patch embeddings
            if reuse_patch_emb:
                assert conv_stem=='none' # no patch embedding if a convolutional stem is used
                print('Reusing patch embeddings.')

                assert old_state_dict['encoder.patch_embed.proj.bias'].shape == pretrained_state_dict['encoder.patch_embed.proj.bias'].shape
                old_state_dict['encoder.patch_embed.proj.bias'] = pretrained_state_dict['encoder.patch_embed.proj.bias']

                _, _, gs_new_h, gs_new_w = old_state_dict['encoder.patch_embed.proj.weight'].shape
                reshaped_weight = adapt_input_conv(in_channels, pretrained_state_dict['encoder.patch_embed.proj.weight'])
                reshaped_weight = F.interpolate(reshaped_weight, size=(gs_new_h, gs_new_w), mode='bilinear')
                pretrained_state_dict['encoder.patch_embed.proj.weight'] = reshaped_weight
            else:
                del pretrained_state_dict['encoder.patch_embed.proj.weight'] # remove patch embedding layers
                del pretrained_state_dict['encoder.patch_embed.proj.bias'] # remove patch embedding layers

            # Delete the pre-trained weights of the decoder
            decoder_keys = []
            for key in pretrained_state_dict.keys():
                if 'decoder' in key:
                    decoder_keys.append(key)
            for decoder_key in decoder_keys:
                del pretrained_state_dict[decoder_key]

            msg = self.rangevit.load_state_dict(pretrained_state_dict, strict=False)
            # The conv stem, decoder and prompt modules are trained from scratch
            num_loaded = len(pretrained_state_dict) - len(msg.unexpected_keys)
            print(f'Loaded {num_loaded} pre-trained encoder tensors '
                  f'(ignored: {", ".join(msg.unexpected_keys) or "none"})')

    def counter_model_parameters(self):
        stats = {}
        stats['total_num_parameters'] = count_parameters(self.rangevit)
        stats['decoder_num_parameters'] = count_parameters(self.rangevit.decoder)
        stats['stem_num_parameters'] = count_parameters(self.rangevit.encoder.patch_embed)
        stats['encoder_num_parameters'] = count_parameters(self.rangevit.encoder) - stats['stem_num_parameters']
        return stats

    def forward(self, input_feature, point_coords=None, point_labels=None):
        """
        input_feature: (B, 5, H, W) range image crop.
        point_coords: (B, N, 2) prompt pixel coordinates as (row, col); -1 marks padding.
        point_labels: (B, N) prompt labels, 1 = foreground, 0 = background, -1 = padding.
        Returns (B, n_cls, H, W) logits.
        """
        image_embeddings, skip = self.rangevit.encoder(input_feature)
        image_embeddings = image_embeddings[:, 1:]  # remove the CLS token

        prompt_embeddings = None
        if point_coords is not None and point_labels is not None:
            valid_points_mask = (point_coords[..., 0] != -1)
            prompt_embeddings = self.prompt_encoder(point_coords, point_labels)
            prompt_embeddings = prompt_embeddings * valid_points_mask.unsqueeze(-1)

        image_pe = self.rangevit.encoder.pos_embed[:, 1:]  # positional encodings without the CLS token

        output = self.rangevit.decoder(image_embeddings, input_feature.shape[-2:], skip, prompt_embeddings, image_pe)

        if self.label_enforcement and prompt_embeddings is not None:
            output = self.enforce_prompt_labels(output, point_coords, point_labels)

        return output

    @staticmethod
    def enforce_prompt_labels(output, point_coords, point_labels):
        """Prompt label enforcement: set the logits at every prompted pixel to +inf for the
        prompt's class and -inf for the other class."""
        B = output.shape[0]
        for b in range(B):
            for coord, label in zip(point_coords[b], point_labels[b]):
                y, x = coord
                if y >= 0 and x >= 0 and label >= 0:
                    output[b, :, y, x] = torch.full_like(output[b, :, y, x], -1e9)
                    output[b, label, y, x] = 1e9
        return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
