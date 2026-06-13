"""Graph-level pooling operators including DiffPool."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_ml.models.gcn import GCN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global pooling functions
# ---------------------------------------------------------------------------


def global_mean_pool(
    x: torch.Tensor, batch: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean pooling over nodes to produce graph-level features.

    Args:
        x: Node features ``(N, D)``.
        batch: Graph membership indices ``(N,)``.  If ``None``, all nodes
            belong to a single graph.

    Returns:
        Graph-level features ``(n_graphs, D)``.
    """
    if batch is None:
        return x.mean(dim=0, keepdim=True)

    n_graphs = int(batch.max().item()) + 1
    out = torch.zeros(n_graphs, x.shape[-1], device=x.device, dtype=x.dtype)
    out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
    counts = torch.bincount(batch, minlength=n_graphs).float().clamp(min=1.0)
    return out / counts.unsqueeze(-1)


def global_max_pool(
    x: torch.Tensor, batch: torch.Tensor | None = None
) -> torch.Tensor:
    """Element-wise max pooling over nodes to produce graph-level features.

    Args:
        x: Node features ``(N, D)``.
        batch: Graph membership indices ``(N,)``.  If ``None``, all nodes
            belong to a single graph.

    Returns:
        Graph-level features ``(n_graphs, D)``.
    """
    if batch is None:
        return x.max(dim=0, keepdim=True).values

    n_graphs = int(batch.max().item()) + 1
    out = torch.full(
        (n_graphs, x.shape[-1]), float("-inf"), device=x.device, dtype=x.dtype
    )
    out.scatter_reduce_(
        0, batch.unsqueeze(-1).expand_as(x), x, reduce="amax", include_self=False
    )
    out = out.masked_fill(out == float("-inf"), 0.0)
    return out


def global_add_pool(
    x: torch.Tensor, batch: torch.Tensor | None = None
) -> torch.Tensor:
    """Sum pooling over nodes to produce graph-level features.

    Args:
        x: Node features ``(N, D)``.
        batch: Graph membership indices ``(N,)``.  If ``None``, all nodes
            belong to a single graph.

    Returns:
        Graph-level features ``(n_graphs, D)``.
    """
    if batch is None:
        return x.sum(dim=0, keepdim=True)

    n_graphs = int(batch.max().item()) + 1
    out = torch.zeros(n_graphs, x.shape[-1], device=x.device, dtype=x.dtype)
    out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
    return out


# ---------------------------------------------------------------------------
# DiffPool: Differentiable Pooling (Ying et al., 2018)
# ---------------------------------------------------------------------------


class DiffPool(nn.Module):
    """Differentiable graph pooling module.

    Learns a soft assignment matrix *S* = softmax(GNN_pool(A, X)) that maps
    *N* nodes to *K* clusters, then coarsens both the feature matrix and the
    adjacency:

        S = softmax(GNN_pool(A, X))        — assignment ``(N, K)``
        X' = S^T X                          — coarsened features ``(K, D)``
        A' = S^T A S                        — coarsened adjacency ``(K, K)``

    Two auxiliary losses encourage meaningful assignments:

    * **Link prediction loss** (LP):
        ||A - S S^T||_F^2  — reconstructed adjacency should match input.
    * **Entropy loss** (E):
        H(S) = -(1/N) Σ_i Σ_k s_{ik} log(s_{ik})  — assignments should be
        crisp (low entropy).

    Args:
        in_dim: Input node feature dimensionality.
        hidden_dim: Hidden width for the embedding and pooling GNNs.
        out_dim: Output feature dimensionality after pooling.
        n_clusters: Number of clusters *K* to pool into.
        n_gnn_layers: Depth of embedding and pooling GNNs.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        n_clusters: int = 10,
        n_gnn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters

        self.embed_gnn = GCN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=n_gnn_layers,
        )
        self.pool_gnn = GCN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=n_clusters,
            n_layers=n_gnn_layers,
        )

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute one DiffPool coarsening step.

        Args:
            x: Node features ``(N, in_dim)``.
            adj: Adjacency matrix ``(N, N)``.

        Returns:
            Tuple of:
                - ``x_pool``: Coarsened features ``(K, out_dim)``.
                - ``adj_pool``: Coarsened adjacency ``(K, K)``.
                - ``lp_loss``: Link prediction reconstruction loss (scalar).
                - ``entropy_loss``: Assignment entropy loss (scalar).
        """
        z = self.embed_gnn(x, adj)
        s_logits = self.pool_gnn(x, adj)
        s = F.softmax(s_logits, dim=-1)  # (N, K)

        x_pool = s.T @ z          # (K, out_dim)
        adj_pool = s.T @ adj @ s  # (K, K)

        lp_loss = self._link_prediction_loss(adj, s)
        entropy_loss = self._entropy_loss(s)

        return x_pool, adj_pool, lp_loss, entropy_loss

    @staticmethod
    def _link_prediction_loss(adj: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Frobenius-norm reconstruction loss: ||A - S S^T||_F^2.

        Args:
            adj: Ground-truth adjacency ``(N, N)``.
            s: Soft assignment ``(N, K)``.

        Returns:
            Scalar loss tensor.
        """
        adj_approx = s @ s.T
        return torch.norm(adj - adj_approx, p="fro") ** 2

    @staticmethod
    def _entropy_loss(s: torch.Tensor) -> torch.Tensor:
        """Mean per-node assignment entropy: -(1/N) Σ_i Σ_k s_{ik} log(s_{ik}).

        Args:
            s: Soft assignment ``(N, K)``.

        Returns:
            Scalar loss tensor.
        """
        eps = 1e-8
        return -(s * torch.log(s + eps)).sum(dim=-1).mean()
