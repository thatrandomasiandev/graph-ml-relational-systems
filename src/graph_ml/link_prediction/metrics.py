"""Link prediction evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class LinkPredictionMetrics:
    auc: float
    ap: float
    hits_at_k: float


def hits_at_k(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> float:
    """Fraction of positive edges ranked in the top-k among all candidates."""
    if len(scores) == 0:
        return 0.0
    k = min(k, len(scores))
    top_idx = np.argpartition(-scores, k - 1)[:k]
    return float(np.any(labels[top_idx] == 1))


def evaluate_link_prediction(
    scores: np.ndarray,
    labels: np.ndarray,
    hits_k: int = 10,
) -> LinkPredictionMetrics:
    labels = labels.astype(int)
    if len(np.unique(labels)) < 2:
        auc = 0.5
        ap = float(np.mean(labels))
    else:
        auc = float(roc_auc_score(labels, scores))
        ap = float(average_precision_score(labels, scores))
    return LinkPredictionMetrics(
        auc=auc,
        ap=ap,
        hits_at_k=hits_at_k(scores, labels, hits_k),
    )
