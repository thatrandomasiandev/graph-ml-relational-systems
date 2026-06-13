"""Tests for temporal graph forecasting."""

from graph_ml.data.temporal_graph_dgp import TemporalDGPConfig, generate_temporal_graph
from graph_ml.temporal.rolling_gnn import RollingTrainConfig, fit_rolling_gcn


def test_rolling_gcn_forecast_metrics():
    data = generate_temporal_graph(
        TemporalDGPConfig(n_nodes=80, n_snapshots=10, seed=20)
    )
    result = fit_rolling_gcn(
        data,
        RollingTrainConfig(epochs=25, hidden_dim=32, history_window=2, seed=20),
    )
    assert 0.0 <= result.forecast_metrics.auc <= 1.0
    assert 0.0 <= result.forecast_metrics.ap <= 1.0
    assert result.train_loss >= 0.0


def test_rolling_gcn_beats_chance():
    data = generate_temporal_graph(
        TemporalDGPConfig(n_nodes=100, n_snapshots=12, drift_strength=0.2, seed=21)
    )
    result = fit_rolling_gcn(
        data,
        RollingTrainConfig(epochs=35, hidden_dim=32, history_window=3, seed=21),
    )
    assert result.forecast_metrics.auc > 0.52
