"""Link prediction training loop for GNN encoders."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from graph_ml.data.base import EdgeSplit, GraphDataset
from graph_ml.link_prediction.metrics import LinkPredictionMetrics, evaluate_link_prediction
from graph_ml.models.gcn import GCN
from graph_ml.models.graphsage import GraphSAGE
from graph_ml.models.link_predictor import dot_product_scores
from graph_ml.utils.seed import set_torch_seed

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def negative_sample(
    n_nodes: int,
    positive_edges: np.ndarray,
    n_negatives: int,
    seed: int = 42,
) -> np.ndarray:
    """Uniform random negative sampling excluding known positives.

    Samples node pairs uniformly at random, rejecting any pair that appears
    in ``positive_edges`` or is a self-loop.

    Args:
        n_nodes: Total number of nodes in the graph.
        positive_edges: Array of positive edges ``(E, 2)`` to exclude.
        n_negatives: Number of negative edges to sample.
        seed: Random seed for reproducibility.

    Returns:
        Array of negative edges ``(n_negatives, 2)`` as int64.
    """
    rng = np.random.default_rng(seed)
    forbidden: set[tuple[int, int]] = set()
    for u, v in positive_edges:
        forbidden.add((min(int(u), int(v)), max(int(u), int(v))))

    negatives: list[tuple[int, int]] = []
    attempts = 0
    max_attempts = n_negatives * 100
    while len(negatives) < n_negatives and attempts < max_attempts:
        u = int(rng.integers(0, n_nodes))
        v = int(rng.integers(0, n_nodes))
        if u == v:
            attempts += 1
            continue
        key = (min(u, v), max(u, v))
        if key not in forbidden:
            negatives.append(key)
            forbidden.add(key)
        attempts += 1

    if len(negatives) < n_negatives:
        logger.warning(
            "Could only sample %d/%d negatives after %d attempts",
            len(negatives), n_negatives, max_attempts,
        )

    return np.asarray(negatives, dtype=np.int64)


def bipartite_split(
    edges: np.ndarray,
    timestamps: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Temporal (chronological) split of edges into train / val / test.

    Sorts edges by timestamp and partitions them sequentially, which
    prevents information leakage from future interactions.

    Args:
        edges: Edge array of shape ``(E, 2)``.
        timestamps: Corresponding timestamps ``(E,)``.
        train_ratio: Fraction of edges for training.
        val_ratio: Fraction of edges for validation.

    Returns:
        Tuple of ``(train_edges, val_edges, test_edges)``, each ``(*, 2)``.

    Raises:
        ValueError: If ratios sum to more than 1.0.
    """
    if train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")

    sort_idx = np.argsort(timestamps)
    sorted_edges = edges[sort_idx]

    n = len(sorted_edges)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = sorted_edges[:n_train]
    val = sorted_edges[n_train : n_train + n_val]
    test = sorted_edges[n_train + n_val :]

    logger.info(
        "Temporal split: %d train / %d val / %d test edges",
        len(train), len(val), len(test),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# LinkPredictionTrainer: class-based training interface
# ---------------------------------------------------------------------------


@dataclass
class LPTrainConfig:
    """Configuration for :class:`LinkPredictionTrainer`.

    Args:
        epochs: Number of training epochs.
        lr: Learning rate.
        hidden_dim: Encoder hidden dimensionality.
        n_layers: Number of encoder message-passing layers.
        dropout: Dropout probability.
        neg_ratio: Negatives per positive during training.
        hits_k: *k* for Hits@k evaluation.
        seed: Random seed.
    """

    epochs: int = 100
    lr: float = 0.005
    hidden_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.2
    neg_ratio: float = 1.0
    hits_k: int = 10
    seed: int = 42


@dataclass
class LPEpochResult:
    """Metrics for a single training epoch.

    Args:
        epoch: Zero-indexed epoch number.
        train_loss: Mean BCE loss over the epoch.
        val_auc: Validation AUC-ROC (``-1.0`` if not computed).
        val_ap: Validation average precision (``-1.0`` if not computed).
    """

    epoch: int
    train_loss: float
    val_auc: float = -1.0
    val_ap: float = -1.0


@dataclass
class LPTrainerResult:
    """Full result container for :class:`LinkPredictionTrainer`.

    Args:
        model_name: Encoder architecture name.
        epoch_results: Per-epoch training metrics.
        test_auc: Test AUC-ROC.
        test_ap: Test average precision.
        test_mrr: Test mean reciprocal rank.
    """

    model_name: str
    epoch_results: list[LPEpochResult]
    test_auc: float
    test_ap: float
    test_mrr: float


class LinkPredictionTrainer:
    """Full-lifecycle link prediction trainer with AUC, AP, and MRR evaluation.

    Wraps a GNN encoder and dot-product decoder, providing ``train_epoch()``
    and ``evaluate()`` methods.

    Args:
        encoder: Any ``nn.Module`` with signature ``forward(x, adj) -> embeddings``.
        dataset: :class:`GraphDataset` with train edges and val/test splits.
        config: :class:`LPTrainConfig` hyperparameters.
    """

    def __init__(
        self,
        encoder: nn.Module,
        dataset: GraphDataset,
        config: LPTrainConfig | None = None,
    ) -> None:
        self.cfg = config or LPTrainConfig()
        set_torch_seed(self.cfg.seed)
        self._rng = np.random.default_rng(self.cfg.seed)

        self.encoder = encoder
        self.dataset = dataset

        self._x = torch.as_tensor(dataset.node_features, dtype=torch.float32)
        self._adj = torch.as_tensor(dataset.adjacency, dtype=torch.float32)

        forbidden = {
            (min(int(u), int(v)), max(int(u), int(v)))
            for u, v in dataset.train_edges
        }
        train_neg = _sample_train_negatives(
            dataset.n_nodes, len(dataset.train_edges), forbidden,
            self.cfg.neg_ratio, self._rng,
        )
        self._train_edges = torch.as_tensor(
            np.vstack([dataset.train_edges, train_neg]), dtype=torch.long
        )
        self._train_labels = torch.as_tensor(
            np.concatenate([
                np.ones(len(dataset.train_edges)),
                np.zeros(len(train_neg)),
            ]),
            dtype=torch.float32,
        )

        self._optimizer = torch.optim.Adam(encoder.parameters(), lr=self.cfg.lr)
        self._criterion = nn.BCEWithLogitsLoss()
        self._epoch_results: list[LPEpochResult] = []

    def train_epoch(self, epoch: int = 0) -> LPEpochResult:
        """Run one training epoch.

        Args:
            epoch: Epoch index (for logging / result tracking).

        Returns:
            :class:`LPEpochResult` with training loss and optional val metrics.
        """
        self.encoder.train()
        self._optimizer.zero_grad()
        emb = self.encoder(self._x, self._adj)
        logits = dot_product_scores(emb, self._train_edges)
        loss = self._criterion(logits, self._train_labels)
        loss.backward()
        self._optimizer.step()

        val_auc, val_ap = -1.0, -1.0
        if self.dataset.val_split.n_edges > 0:
            val_scores = self._score_edges(self.dataset.val_split.edges)
            labels = self.dataset.val_split.labels.astype(int)
            if len(np.unique(labels)) >= 2:
                val_auc = float(roc_auc_score(labels, val_scores))
                val_ap = float(average_precision_score(labels, val_scores))

        result = LPEpochResult(
            epoch=epoch,
            train_loss=float(loss.item()),
            val_auc=val_auc,
            val_ap=val_ap,
        )
        self._epoch_results.append(result)
        return result

    def evaluate(self) -> LPTrainerResult:
        """Evaluate trained model on the test split.

        Computes AUC-ROC, average precision, and mean reciprocal rank (MRR).

        Returns:
            :class:`LPTrainerResult` with all metrics.
        """
        self.encoder.eval()
        test_edges = self.dataset.test_split.edges
        test_labels = self.dataset.test_split.labels.astype(int)
        test_scores = self._score_edges(test_edges)

        if len(np.unique(test_labels)) >= 2:
            test_auc = float(roc_auc_score(test_labels, test_scores))
            test_ap = float(average_precision_score(test_labels, test_scores))
        else:
            test_auc = 0.5
            test_ap = float(np.mean(test_labels))

        test_mrr = self._compute_mrr(test_scores, test_labels)

        return LPTrainerResult(
            model_name=type(self.encoder).__name__,
            epoch_results=list(self._epoch_results),
            test_auc=test_auc,
            test_ap=test_ap,
            test_mrr=test_mrr,
        )

    def fit(self) -> LPTrainerResult:
        """Convenience method: run all epochs then evaluate.

        Returns:
            :class:`LPTrainerResult` from :meth:`evaluate`.
        """
        for epoch in range(self.cfg.epochs):
            result = self.train_epoch(epoch)
            if epoch % 20 == 0:
                logger.info(
                    "Epoch %d — loss=%.4f  val_auc=%.4f",
                    epoch, result.train_loss, result.val_auc,
                )
        return self.evaluate()

    def _score_edges(self, edges: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            emb = self.encoder(self._x, self._adj)
            edge_t = torch.as_tensor(edges, dtype=torch.long, device=self._x.device)
            return dot_product_scores(emb, edge_t).cpu().numpy()

    @staticmethod
    def _compute_mrr(scores: np.ndarray, labels: np.ndarray) -> float:
        """Mean reciprocal rank of positive edges.

        Args:
            scores: Predicted scores ``(E,)``.
            labels: Binary labels ``(E,)``.

        Returns:
            MRR value in ``[0, 1]``.
        """
        pos_mask = labels == 1
        if not np.any(pos_mask):
            return 0.0

        sorted_idx = np.argsort(-scores)
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[sorted_idx] = np.arange(1, len(scores) + 1)

        pos_ranks = ranks[pos_mask]
        return float(np.mean(1.0 / pos_ranks))
