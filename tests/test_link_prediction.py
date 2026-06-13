"""Tests for link prediction models and metrics."""

import numpy as np
import torch

from graph_ml.data.static_graph_dgp import SBMDGPConfig, generate_sbm_graph
from graph_ml.link_prediction.metrics import evaluate_link_prediction
from graph_ml.link_prediction.trainer import TrainConfig, fit_link_predictor
from graph_ml.models.gcn import GCN
from graph_ml.models.graphsage import GraphSAGE


def test_perfect_scores_metrics():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([10.0, 9.0, -5.0, -6.0])
    metrics = evaluate_link_prediction(scores, labels, hits_k=2)
    assert metrics.auc == 1.0
    assert metrics.ap == 1.0
    assert metrics.hits_at_k == 1.0


def test_gcn_forward_shape():
    x = torch.randn(20, 8)
    adj = torch.eye(20)
    model = GCN(8, hidden_dim=16, out_dim=16, n_layers=2)
    out = model(x, adj)
    assert out.shape == (20, 16)


def test_graphsage_forward_shape():
    x = torch.randn(20, 8)
    adj = torch.eye(20)
    model = GraphSAGE(8, hidden_dim=16, out_dim=16, n_layers=2)
    out = model(x, adj)
    assert out.shape == (20, 16)


def test_fit_link_predictor_runs():
    data = generate_sbm_graph(SBMDGPConfig(n_nodes=80, seed=5))
    result = fit_link_predictor(
        data,
        model_name="GCN",
        config=TrainConfig(epochs=5, seed=5),
    )
    assert result.model_name == "GCN"
    assert 0.0 <= result.test_metrics.auc <= 1.0
    assert 0.0 <= result.test_metrics.ap <= 1.0
    assert result.test_metrics.hits_at_k in (0.0, 1.0)
