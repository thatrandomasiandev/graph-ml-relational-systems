"""Temporal graph DGP with drifting latent positions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graph_ml.data.base import EdgeSplit, TemporalGraphDataset
from graph_ml.utils.seed import set_seed


@dataclass
class TemporalDGPConfig:
    n_nodes: int = 150
    n_snapshots: int = 12
    feature_dim: int = 12
    drift_strength: float = 0.15
    k_neighbors: int = 8
    feature_noise: float = 0.25
    neg_ratio: float = 1.0
    seed: int = 42


def _knn_adjacency(positions: np.ndarray, k_neighbors: int) -> np.ndarray:
    """Connect each node to its k nearest neighbors under RBF similarity."""
    n = positions.shape[0]
    k = min(k_neighbors, n - 1)
    if k < 1:
        raise ValueError("k_neighbors must be >= 1 and n_nodes must be >= 2.")

    diff = positions[:, None, :] - positions[None, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    sim = np.exp(-dist_sq)
    np.fill_diagonal(sim, 0.0)

    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        neighbors = np.argpartition(-sim[i], k - 1)[:k]
        for j in neighbors:
            if sim[i, j] > 0:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    return adj


def _snapshot_edges(adj: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(adj.shape[0], k=1)
    mask = adj[idx] > 0
    return np.column_stack([idx[0][mask], idx[1][mask]]).astype(np.int64)


def _sample_negatives(
    n_nodes: int,
    n_pos: int,
    forbidden: set[tuple[int, int]],
    neg_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_neg = max(1, int(n_pos * neg_ratio))
    negatives: list[tuple[int, int]] = []
    attempts = 0
    max_attempts = max(n_neg * 100, 500)
    while len(negatives) < n_neg and attempts < max_attempts:
        u = int(rng.integers(0, n_nodes))
        v = int(rng.integers(0, n_nodes))
        if u == v:
            attempts += 1
            continue
        key = (min(u, v), max(u, v))
        if key in forbidden:
            attempts += 1
            continue
        negatives.append(key)
        attempts += 1
    return np.asarray(negatives, dtype=np.int64)


def generate_temporal_graph(config: TemporalDGPConfig | None = None) -> TemporalGraphDataset:
    """
    Generate a temporal graph sequence from drifting latent positions.

    DGP:
      - Initial latent positions Z_0 ~ N(0, I)
      - Z_t = Z_{t-1} + drift_strength * eps_t
      - A_t: k-NN graph on RBF similarity of Z_t
      - Forecast target: held-out edges from the final snapshot
    """
    cfg = config or TemporalDGPConfig()
    rng = set_seed(cfg.seed)

    positions = rng.standard_normal((cfg.n_nodes, cfg.feature_dim)).astype(np.float32)
    snapshots: list[np.ndarray] = []
    latent_trajectory: list[np.ndarray] = [positions.copy()]

    for _ in range(cfg.n_snapshots):
        positions = positions + cfg.drift_strength * rng.standard_normal(positions.shape)
        positions = positions.astype(np.float32)
        latent_trajectory.append(positions.copy())
        snapshots.append(_knn_adjacency(positions, cfg.k_neighbors))

    final_positions = latent_trajectory[-1]
    node_features = final_positions + cfg.feature_noise * rng.standard_normal(
        (cfg.n_nodes, cfg.feature_dim)
    )
    node_features = node_features.astype(np.float32)

    prev_adj = snapshots[-2]
    next_adj = snapshots[-1]

    pos_edges = _snapshot_edges(next_adj)
    rng.shuffle(pos_edges)
    n_pos = min(len(pos_edges), max(20, len(pos_edges) // 2))
    pos_edges = pos_edges[:n_pos]

    existing = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in pos_edges}
    neg_arr = _sample_negatives(cfg.n_nodes, len(pos_edges), existing, cfg.neg_ratio, rng)
    if len(neg_arr) == 0:
        raise ValueError("Could not sample negative edges for temporal forecast split.")

    forecast = EdgeSplit(
        edges=np.vstack([pos_edges, neg_arr]),
        labels=np.concatenate([np.ones(len(pos_edges)), np.zeros(len(neg_arr))]),
    )

    return TemporalGraphDataset(
        node_features=node_features,
        snapshots=snapshots,
        forecast_split=forecast,
        metadata={
            "dgp": "drifting_latent_knn",
            "n_nodes": cfg.n_nodes,
            "n_snapshots": cfg.n_snapshots,
            "feature_dim": cfg.feature_dim,
            "drift_strength": cfg.drift_strength,
            "k_neighbors": cfg.k_neighbors,
            "seed": cfg.seed,
        },
        ground_truth={
            "latent_trajectory": latent_trajectory,
            "final_adjacency": next_adj,
            "prev_adjacency": prev_adj,
            "final_positions": final_positions,
        },
    )
