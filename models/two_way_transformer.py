# Two-way transformer adapted from Segment Anything (Kirillov et al., 2023)
# https://github.com/facebookresearch/segment-anything
# with a learnable scaling of the prompt attention added for RangeSAM.

import math
from typing import Tuple, Type
import torch
import torch.nn as nn
from torch import Tensor

class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        learnable_prompt_scale: bool = True,
    ):
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.layers = nn.ModuleList()

        for _ in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    learnable_prompt_scale=learnable_prompt_scale,
                )
            )

        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, learnable_prompt_scale)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Prepare queries and keys
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from points to image
        attn_out = self.final_attn_token_to_image(q=queries, k=keys, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys

class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        learnable_prompt_scale: bool = True,
    ):
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads, learnable_prompt_scale)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, learnable_prompt_scale)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, learnable_prompt_scale)

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention for queries
        q = queries + query_pe
        attn_out = self.self_attn(q=q, k=q, v=queries, is_prompt = True)
        queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention: tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys, is_prompt = True)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention: image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries, is_prompt = False)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, learnable_prompt_scale: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        # Scaling factor balancing prompt and image information in the prompt attentions.
        # Fixed to 1 when disabled (kept as a buffer so checkpoints stay compatible).
        if learnable_prompt_scale:
            self.prompt_scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('prompt_scale', torch.ones(1))

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, is_prompt=False) -> Tensor:
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)

        if is_prompt:
            # Apply scaling only to prompt embeddings' projections
            q = q * self.prompt_scale
            k = k * self.prompt_scale

        q, k, v = map(lambda x: self._separate_heads(x, self.num_heads), (q, k, v))

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
