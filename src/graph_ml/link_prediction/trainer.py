"""Link prediction training loop for GNN encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from graph_ml.data.base import GraphDataset
from graph_ml.link_prediction.metrics import LinkPredictionMetrics, evaluate_link_prediction
from graph_ml.models.gcn import GCN
from graph_ml.models.graphsage import GraphSAGE
from graph_ml.models.link_predictor import dot_product_scores
from graph_ml.utils.seed import set_torch_seed


@dataclass
class TrainConfig:
    epochs: int = 80
    lr: float = 0.01
    hidden_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.2
    neg_ratio: float = 1.0
    hits_k: int = 10
    seed: int = 42


@dataclass
class LinkPredictionResult:
    model_name: str
    train_loss: float
    val_metrics: LinkPredictionMetrics
    test_metrics: LinkPredictionMetrics
    score_fn: Callable[[np.ndarray], np.ndarray]


def _build_encoder(name: str, in_dim: int, cfg: TrainConfig) -> nn.Module:
    if name == "GCN":
        return GCN(in_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.n_layers, cfg.dropout)
    if name == "GraphSAGE":
        return GraphSAGE(in_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.n_layers, cfg.dropout)
    raise ValueError(f"Unknown model: {name}")


def _sample_train_negatives(
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


def _predict_scores(
    encoder: nn.Module,
    x: torch.Tensor,
    adj: torch.Tensor,
    edges: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        emb = encoder(x, adj)
        edge_t = torch.as_tensor(edges, dtype=torch.long, device=x.device)
        scores = dot_product_scores(emb, edge_t).cpu().numpy()
    return scores


def fit_link_predictor(
    dataset: GraphDataset,
    model_name: str = "GCN",
    config: TrainConfig | None = None,
) -> LinkPredictionResult:
    """Train a GNN encoder with dot-product link prediction loss."""
    cfg = config or TrainConfig()
    set_torch_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    x = torch.as_tensor(dataset.node_features, dtype=torch.float32)
    adj = torch.as_tensor(dataset.adjacency, dtype=torch.float32)
    encoder = _build_encoder(model_name, dataset.feature_dim, cfg)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.lr)
    criterion = nn.BCEWithLogitsLoss()

    forbidden = {
        (min(int(u), int(v)), max(int(u), int(v))) for u, v in dataset.train_edges
    }
    train_pos = dataset.train_edges
    train_neg = _sample_train_negatives(
        dataset.n_nodes, len(train_pos), forbidden, cfg.neg_ratio, rng
    )
    train_edges = np.vstack([train_pos, train_neg])
    train_labels = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])

    edge_t = torch.as_tensor(train_edges, dtype=torch.long)
    label_t = torch.as_tensor(train_labels, dtype=torch.float32)

    encoder.train()
    last_loss = 0.0
    for _ in range(cfg.epochs):
        optimizer.zero_grad()
        emb = encoder(x, adj)
        logits = dot_product_scores(emb, edge_t)
        loss = criterion(logits, label_t)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

    encoder.eval()
    val_scores = _predict_scores(encoder, x, adj, dataset.val_split.edges)
    test_scores = _predict_scores(encoder, x, adj, dataset.test_split.edges)

    val_metrics = evaluate_link_prediction(
        val_scores, dataset.val_split.labels, hits_k=cfg.hits_k
    )
    test_metrics = evaluate_link_prediction(
        test_scores, dataset.test_split.labels, hits_k=cfg.hits_k
    )

    def score_fn(edges: np.ndarray) -> np.ndarray:
        return _predict_scores(encoder, x, adj, edges)

    return LinkPredictionResult(
        model_name=model_name,
        train_loss=last_loss,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        score_fn=score_fn,
    )
