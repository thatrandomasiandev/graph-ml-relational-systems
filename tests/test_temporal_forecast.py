"""Tests for temporal graph forecasting."""

import numpy as np

from graph_ml.data.temporal_graph_dgp import TemporalDGPConfig, generate_temporal_graph
from graph_ml.temporal.metrics import evaluate_temporal_forecast
from graph_ml.temporal.rolling_gnn import RollingTrainConfig, fit_rolling_gcn


def test_temporal_metrics_sanity():
    labels = np.array([1, 0, 1, 0])
    scores = np.array([2.0, -1.0, 1.5, -2.0])
    metrics = evaluate_temporal_forecast(scores, labels, threshold=0.0)
    assert metrics.auc == 1.0
    assert metrics.edge_f1 == 1.0


def test_fit_rolling_gcn_runs():
    data = generate_temporal_graph(
        TemporalDGPConfig(n_nodes=60, n_snapshots=8, seed=6)
    )
    result = fit_rolling_gcn(
        data,
        config=RollingTrainConfig(epochs=5, history_window=2, seed=6),
    )
    assert result.model_name == "RollingGCN"
    assert 0.0 <= result.forecast_metrics.auc <= 1.0
    assert 0.0 <= result.forecast_metrics.ap <= 1.0
    assert 0.0 <= result.forecast_metrics.edge_f1 <= 1.0
