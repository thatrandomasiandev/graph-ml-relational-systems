"""Dataset containers for static and temporal graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EdgeSplit:
    """Positive and negative edge pairs for link prediction."""

    edges: np.ndarray
    labels: np.ndarray

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])


@dataclass
class GraphDataset:
    """Static attributed graph with train/val/test link splits."""

    node_features: np.ndarray
    train_edges: np.ndarray
    val_split: EdgeSplit
    test_split: EdgeSplit
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.node_features.shape[1])

    @property
    def adjacency(self) -> np.ndarray:
        """Undirected adjacency from training edges only (transductive setting)."""
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        for u, v in self.train_edges:
            adj[int(u), int(v)] = 1.0
            adj[int(v), int(u)] = 1.0
        return adj


@dataclass
class TemporalGraphDataset:
    """Sequence of graph snapshots with next-step prediction targets."""

    node_features: np.ndarray
    snapshots: list[np.ndarray]
    forecast_split: EdgeSplit
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def n_snapshots(self) -> int:
        return len(self.snapshots)

    @property
    def feature_dim(self) -> int:
        return int(self.node_features.shape[1])
