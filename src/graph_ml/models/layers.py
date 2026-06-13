"""Graph convolution and propagation utilities."""

from __future__ import annotations

import torch


def normalized_adjacency(adj: torch.Tensor, self_loops: bool = True) -> torch.Tensor:
    """Compute D^{-1/2} (A + I) D^{-1/2} for symmetric normalized propagation."""
    a = adj.clone()
    if self_loops:
        a = a + torch.eye(a.shape[0], device=a.device, dtype=a.dtype)
    deg = a.sum(dim=1).clamp(min=1.0)
    inv_sqrt = deg.pow(-0.5)
    return inv_sqrt[:, None] * a * inv_sqrt[None, :]


def propagate(adj_norm: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """One-hop mean aggregation: A_norm @ X."""
    return adj_norm @ features
