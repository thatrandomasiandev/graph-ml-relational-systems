"""Temporal link forecasting metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


@dataclass
class TemporalMetrics:
    auc: float
    ap: float
    edge_f1: float


def evaluate_temporal_forecast(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.0,
) -> TemporalMetrics:
    labels = labels.astype(int)
    if len(np.unique(labels)) < 2:
        auc = 0.5
        ap = float(np.mean(labels)) if len(labels) else 0.0
    else:
        auc = float(roc_auc_score(labels, scores))
        ap = float(average_precision_score(labels, scores))

    preds = (scores >= threshold).astype(int)
    edge_f1 = float(f1_score(labels, preds, zero_division=0))
    return TemporalMetrics(auc=auc, ap=ap, edge_f1=edge_f1)
