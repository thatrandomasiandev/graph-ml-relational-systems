"""GraphSAGE-style mean aggregator encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from graph_ml.models.layers import normalized_adjacency, propagate


class GraphSAGE(nn.Module):
    """Inductive GraphSAGE with mean neighborhood aggregation."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.self_layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.neigh_layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers)])

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj_norm = normalized_adjacency(adj)
        h = x
        for i in range(self.n_layers):
            neigh = propagate(adj_norm, h)
            h = self.self_layers[i](h) + self.neigh_layers[i](neigh)
            if i < self.n_layers - 1:
                h = torch.relu(h)
                h = self.dropout(h)
        return h
