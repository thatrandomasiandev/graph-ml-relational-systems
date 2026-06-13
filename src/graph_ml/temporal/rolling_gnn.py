"""Rolling-window GCN for temporal link forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from graph_ml.data.base import TemporalGraphDataset
from graph_ml.models.gcn import GCN
from graph_ml.models.link_predictor import dot_product_scores
from graph_ml.temporal.metrics import TemporalMetrics, evaluate_temporal_forecast
from graph_ml.utils.seed import set_torch_seed


@dataclass
class RollingTrainConfig:
    epochs: int = 60
    lr: float = 0.01
    hidden_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.2
    history_window: int = 3
    neg_ratio: float = 1.0
    seed: int = 42


@dataclass
class RollingGCNResult:
    model_name: str
    train_loss: float
    forecast_metrics: TemporalMetrics


def _cumulative_adjacency(snapshots: list[np.ndarray], end_idx: int) -> np.ndarray:
    """Union adjacency over snapshots [0, end_idx]."""
    adj = np.zeros_like(snapshots[0], dtype=np.float32)
    for t in range(end_idx + 1):
        adj = np.maximum(adj, snapshots[t])
    return adj


def _snapshot_pos_edges(adj: np.ndarray) -> np.ndarray:
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
    while len(negatives) < n_neg and attempts < n_neg * 50:
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


def _build_training_pairs(
    dataset: TemporalGraphDataset,
    neg_ratio: float,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Create (adjacency, pos_edges, neg_edges) tuples for each forecast step."""
    pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for t in range(1, dataset.n_snapshots):
        cum_adj = _cumulative_adjacency(dataset.snapshots, t - 1)
        pos = _snapshot_pos_edges(dataset.snapshots[t])
        if len(pos) == 0:
            continue
        n_pos = min(len(pos), 40)
        rng.shuffle(pos)
        pos = pos[:n_pos]
        forbidden = {
            (min(int(u), int(v)), max(int(u), int(v)))
            for snap in dataset.snapshots[: t + 1]
            for u, v in _snapshot_pos_edges(snap)
        }
        neg = _sample_negatives(dataset.n_nodes, len(pos), forbidden, neg_ratio, rng)
        pairs.append((cum_adj, pos, neg))
    return pairs


def fit_rolling_gcn(
    dataset: TemporalGraphDataset,
    config: RollingTrainConfig | None = None,
) -> RollingGCNResult:
    """
    Train a GCN on cumulative snapshots to predict edges at the next timestep.

    For each t, adjacency = union(snapshots[0:t]), target = edges in snapshot t.
    """
    cfg = config or RollingTrainConfig()
    set_torch_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    x = torch.as_tensor(dataset.node_features, dtype=torch.float32)
    encoder = GCN(
        dataset.feature_dim,
        cfg.hidden_dim,
        cfg.hidden_dim,
        cfg.n_layers,
        cfg.dropout,
    )
    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.lr)
    criterion = nn.BCEWithLogitsLoss()

    train_pairs = _build_training_pairs(dataset, cfg.neg_ratio, rng)
    if not train_pairs:
        raise ValueError("No training snapshots available for rolling GCN.")

    encoder.train()
    last_loss = 0.0
    for _ in range(cfg.epochs):
        total_loss = 0.0
        for cum_adj, pos, neg in train_pairs:
            edges = np.vstack([pos, neg])
            labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            adj = torch.as_tensor(cum_adj, dtype=torch.float32)
            edge_t = torch.as_tensor(edges, dtype=torch.long)
            label_t = torch.as_tensor(labels, dtype=torch.float32)

            optimizer.zero_grad()
            emb = encoder(x, adj)
            logits = dot_product_scores(emb, edge_t)
            loss = criterion(logits, label_t)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        last_loss = total_loss / len(train_pairs)

    encoder.eval()
    forecast_adj = _cumulative_adjacency(dataset.snapshots, dataset.n_snapshots - 2)
    adj_t = torch.as_tensor(forecast_adj, dtype=torch.float32)
    edges = dataset.forecast_split.edges
    with torch.no_grad():
        emb = encoder(x, adj_t)
        edge_t = torch.as_tensor(edges, dtype=torch.long)
        scores = dot_product_scores(emb, edge_t).cpu().numpy()

    metrics = evaluate_temporal_forecast(scores, dataset.forecast_split.labels)
    return RollingGCNResult(
        model_name="RollingGCN",
        train_loss=last_loss,
        forecast_metrics=metrics,
    )
