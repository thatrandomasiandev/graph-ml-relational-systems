"""Link scoring functions."""

from __future__ import annotations

import torch
import torch.nn as nn


def dot_product_scores(embeddings: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Score edges via elementwise dot product of endpoint embeddings."""
    src = embeddings[edges[:, 0]]
    dst = embeddings[edges[:, 1]]
    return (src * dst).sum(dim=-1)


class MLPLinkScorer(nn.Module):
    """Concatenate endpoint embeddings and score with an MLP."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        src = embeddings[edges[:, 0]]
        dst = embeddings[edges[:, 1]]
        return self.net(torch.cat([src, dst], dim=-1)).squeeze(-1)


def mlp_link_scores(
    embeddings: torch.Tensor,
    edges: torch.Tensor,
    scorer: MLPLinkScorer,
) -> torch.Tensor:
    return scorer(embeddings, edges)
