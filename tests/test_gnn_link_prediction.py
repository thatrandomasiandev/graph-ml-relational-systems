"""Tests that GNN link predictors beat random on SBM data."""

import numpy as np

from graph_ml.data.static_graph_dgp import SBMDGPConfig, generate_sbm_graph
from graph_ml.link_prediction.trainer import TrainConfig, fit_link_predictor


def test_gcn_beats_random_auc():
    data = generate_sbm_graph(
        SBMDGPConfig(n_nodes=150, p_in=0.3, p_out=0.02, feature_dim=16, seed=10)
    )
    result = fit_link_predictor(
        data,
        model_name="GCN",
        config=TrainConfig(epochs=40, hidden_dim=32, seed=10),
    )
    assert result.test_metrics.auc > 0.55


def test_graphsage_produces_finite_scores():
    data = generate_sbm_graph(SBMDGPConfig(n_nodes=100, seed=11))
    result = fit_link_predictor(
        data,
        model_name="GraphSAGE",
        config=TrainConfig(epochs=30, hidden_dim=32, seed=11),
    )
    scores = result.score_fn(data.test_split.edges)
    assert np.all(np.isfinite(scores))
    assert len(scores) == data.test_split.n_edges
