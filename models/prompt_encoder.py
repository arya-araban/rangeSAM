# Point prompt encoder adapted from Segment Anything (Kirillov et al., 2023)
# https://github.com/facebookresearch/segment-anything

import torch
import torch.nn as nn
import numpy as np

class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: float = 1.0):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.scale = scale
        self.register_buffer("positional_encoding_gaussian_matrix", scale * torch.randn((2, num_pos_feats)))

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward_with_coords(self, coords_input: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))

class PromptEncoder(nn.Module):
    def __init__(self, embed_dim: int, image_size: tuple):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # Learnable embeddings added to the positional encoding of each prompt:
        # index 0 for background prompts, index 1 for foreground prompts.
        self.point_embeddings = nn.ModuleList([nn.Embedding(1, embed_dim) for _ in range(2)])
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        point_embedding = self.pe_layer.forward_with_coords(points, self.image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def forward(self, points: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self._embed_points(points, labels)
