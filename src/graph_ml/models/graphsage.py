"""GraphSAGE variants with pluggable neighborhood aggregators."""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_ml.models.layers import normalized_adjacency, propagate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aggregator functions
# ---------------------------------------------------------------------------


def _mean_aggregator(adj: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Mean aggregation: D^{-1} A h."""
    adj_norm = normalized_adjacency(adj, self_loops=False)
    return adj_norm @ h


def _max_aggregator(adj: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Element-wise max-pooling over neighbor features.

    For each node *i*, computes max over the set {h_j : A_{ij} > 0}.
    Nodes with no neighbors receive zeros.
    """
    mask = (adj > 0).unsqueeze(-1)  # (N, N, 1)
    h_expanded = h.unsqueeze(0).expand(adj.shape[0], -1, -1)  # (N, N, D)
    h_masked = h_expanded.masked_fill(~mask, float("-inf"))
    pooled, _ = h_masked.max(dim=1)  # (N, D)
    pooled = pooled.masked_fill(pooled == float("-inf"), 0.0)
    return pooled


def _lstm_aggregator(adj: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """LSTM aggregation over randomly-permuted neighbor sequences.

    Uses a single-layer LSTM whose hidden-state output for each node is the
    aggregated representation. Neighbors are processed in the order they
    appear in the adjacency row (permutation-invariant in expectation when
    combined with random neighbor sampling during training).
    """
    n_nodes, feat_dim = h.shape
    lstm = nn.LSTM(feat_dim, feat_dim, batch_first=True).to(h.device)
    outputs = torch.zeros_like(h)
    for i in range(n_nodes):
        neighbors = (adj[i] > 0).nonzero(as_tuple=False).squeeze(-1)
        if neighbors.numel() == 0:
            continue
        seq = h[neighbors].unsqueeze(0)  # (1, n_neigh, D)
        _, (h_n, _) = lstm(seq)
        outputs[i] = h_n.squeeze(0).squeeze(0)
    return outputs


SAGE_AGGREGATORS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "mean": _mean_aggregator,
    "max": _max_aggregator,
    "lstm": _lstm_aggregator,
}
"""Registry of neighborhood aggregation functions for GraphSAGE.

Each aggregator has signature ``(adj: Tensor, h: Tensor) -> Tensor``
where *adj* is ``(N, N)`` and *h* is ``(N, D)``.

Available keys: ``"mean"``, ``"max"``, ``"lstm"``.
"""


# ---------------------------------------------------------------------------
# Original transductive GraphSAGE (kept for backward compatibility)
# ---------------------------------------------------------------------------


class GraphSAGE(nn.Module):
    """Transductive GraphSAGE with mean neighborhood aggregation.

    Implements the SAGE update rule:
        h_v^{(l)} = σ(W_self · h_v^{(l-1)} + W_neigh · AGG({h_u^{(l-1)} : u ∈ N(v)}))

    Args:
        in_dim: Input feature dimensionality.
        hidden_dim: Width of hidden layers.
        out_dim: Output embedding dimensionality.
        n_layers: Number of aggregation layers (must be >= 1).
        dropout: Dropout probability between layers.
    """

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
        """Forward pass.

        Args:
            x: Node features ``(N, in_dim)``.
            adj: Adjacency matrix ``(N, N)``.

        Returns:
            Node embeddings ``(N, out_dim)``.
        """
        adj_norm = normalized_adjacency(adj)
        h = x
        for i in range(self.n_layers):
            neigh = propagate(adj_norm, h)
            h = self.self_layers[i](h) + self.neigh_layers[i](neigh)
            if i < self.n_layers - 1:
                h = torch.relu(h)
                h = self.dropout(h)
        return h


# ---------------------------------------------------------------------------
# Inductive GraphSAGE (supports unseen nodes at inference)
# ---------------------------------------------------------------------------


class InductiveGraphSAGE(nn.Module):
    """Inductive GraphSAGE with pluggable aggregator.

    Unlike the transductive variant, this model does *not* depend on a fixed
    node ordering. Given *any* adjacency matrix and feature matrix (including
    graphs with nodes unseen during training), the model can produce
    embeddings.

    Supports ``"mean"``, ``"max"``, and ``"lstm"`` aggregators via the
    :data:`SAGE_AGGREGATORS` registry.

    Update rule per layer:
        h_N(v)   = AGG({h_u^{(l-1)} : u ∈ N(v)})
        h_v^{(l)} = σ(W · CONCAT(h_v^{(l-1)}, h_N(v)))

    followed by L2 normalization of each node embedding.

    Args:
        in_dim: Input feature dimensionality.
        hidden_dim: Hidden layer width.
        out_dim: Output embedding dimensionality.
        n_layers: Number of aggregation layers (must be >= 1).
        dropout: Dropout probability between layers.
        aggregator: Aggregator name (``"mean"``, ``"max"``, or ``"lstm"``).
        normalize: Whether to L2-normalize output embeddings.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        aggregator: str = "mean",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if aggregator not in SAGE_AGGREGATORS:
            raise ValueError(
                f"Unknown aggregator '{aggregator}'. "
                f"Choose from {list(SAGE_AGGREGATORS.keys())}"
            )

        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)
        self.aggregator_name = aggregator
        self._aggregate = SAGE_AGGREGATORS[aggregator]
        self.normalize = normalize

        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.linear_layers = nn.ModuleList([
            nn.Linear(dims[i] * 2, dims[i + 1]) for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass (works on any graph, including unseen nodes).

        Args:
            x: Node features ``(N, in_dim)``.
            adj: Adjacency matrix ``(N, N)``.

        Returns:
            Node embeddings ``(N, out_dim)``, optionally L2-normalized.
        """
        h = x
        for i, linear in enumerate(self.linear_layers):
            neigh = self._aggregate(adj, h)
            h = torch.cat([h, neigh], dim=-1)
            h = linear(h)
            if i < self.n_layers - 1:
                h = F.relu(h)
                h = self.dropout(h)

        if self.normalize:
            h = F.normalize(h, p=2, dim=-1)
        return h
