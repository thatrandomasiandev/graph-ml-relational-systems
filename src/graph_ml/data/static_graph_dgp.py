"""Stochastic block model DGP with latent community features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graph_ml.data.base import EdgeSplit, GraphDataset
from graph_ml.utils.seed import set_seed


@dataclass
class SBMDGPConfig:
    n_nodes: int = 300
    n_communities: int = 4
    feature_dim: int = 16
    p_in: float = 0.25
    p_out: float = 0.02
    feature_noise: float = 0.35
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    neg_ratio: float = 1.0
    seed: int = 42


def _assign_communities(n_nodes: int, n_communities: int, rng: np.random.Generator) -> np.ndarray:
    sizes = np.full(n_communities, n_nodes // n_communities, dtype=int)
    sizes[: n_nodes % n_communities] += 1
    labels = np.repeat(np.arange(n_communities), sizes)
    rng.shuffle(labels)
    return labels


def _sbm_adjacency(
    labels: np.ndarray,
    p_in: float,
    p_out: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(labels)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            prob = p_in if labels[i] == labels[j] else p_out
            if rng.random() < prob:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    return adj


def _upper_triangle_edges(adj: np.ndarray) -> np.ndarray:
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
    n_neg = int(n_pos * neg_ratio)
    negatives: list[tuple[int, int]] = []
    attempts = 0
    max_attempts = n_neg * 50
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
        forbidden.add(key)
        attempts += 1
    return np.asarray(negatives, dtype=np.int64)


def _split_edges(
    edges: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    neg_ratio: float,
    n_nodes: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, EdgeSplit, EdgeSplit]:
    rng.shuffle(edges)
    n = len(edges)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, n - n_train - n_val)

    train = edges[:n_train]
    val_pos = edges[n_train : n_train + n_val]
    test_pos = edges[n_train + n_val : n_train + n_val + n_test]

    all_pos = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in edges}

    val_neg = _sample_negatives(n_nodes, len(val_pos), set(all_pos), neg_ratio, rng)
    test_neg = _sample_negatives(n_nodes, len(test_pos), set(all_pos), neg_ratio, rng)

    val_split = EdgeSplit(
        edges=np.vstack([val_pos, val_neg]),
        labels=np.concatenate([np.ones(len(val_pos)), np.zeros(len(val_neg))]),
    )
    test_split = EdgeSplit(
        edges=np.vstack([test_pos, test_neg]),
        labels=np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))]),
    )
    return train, val_split, test_split


def generate_sbm_graph(config: SBMDGPConfig | None = None) -> GraphDataset:
    """
    Generate an attributed SBM graph with transductive link splits.

    DGP:
      - Nodes assigned to K communities
      - Edge probability p_in within community, p_out across
      - Node features = community centroid + Gaussian noise
    """
    cfg = config or SBMDGPConfig()
    rng = set_seed(cfg.seed)

    labels = _assign_communities(cfg.n_nodes, cfg.n_communities, rng)
    adj = _sbm_adjacency(labels, cfg.p_in, cfg.p_out, rng)
    centroids = rng.standard_normal((cfg.n_communities, cfg.feature_dim))
    features = centroids[labels] + cfg.feature_noise * rng.standard_normal(
        (cfg.n_nodes, cfg.feature_dim)
    )
    features = features.astype(np.float32)
    edges = _upper_triangle_edges(adj)
    if len(edges) == 0:
        raise ValueError("SBM produced an empty graph; increase p_in or n_nodes.")

    train_edges, val_split, test_split = _split_edges(
        edges,
        cfg.train_ratio,
        cfg.val_ratio,
        cfg.neg_ratio,
        cfg.n_nodes,
        rng,
    )

    return GraphDataset(
        node_features=features,
        train_edges=train_edges,
        val_split=val_split,
        test_split=test_split,
        metadata={
            "dgp": "attributed_sbm",
            "n_nodes": cfg.n_nodes,
            "n_communities": cfg.n_communities,
            "feature_dim": cfg.feature_dim,
            "p_in": cfg.p_in,
            "p_out": cfg.p_out,
            "n_train_edges": len(train_edges),
            "seed": cfg.seed,
        },
        ground_truth={
            "labels": labels,
            "adjacency": adj,
            "centroids": centroids,
        },
    )
