"""Tests for synthetic graph DGP invariants."""

import numpy as np

from graph_ml.data.static_graph_dgp import SBMDGPConfig, generate_sbm_graph
from graph_ml.data.temporal_graph_dgp import TemporalDGPConfig, generate_temporal_graph


def test_sbm_shapes_and_ground_truth():
    data = generate_sbm_graph(SBMDGPConfig(n_nodes=120, seed=0))
    assert data.node_features.shape == (120, 16)
    assert data.ground_truth["adjacency"].shape == (120, 120)
    assert len(data.ground_truth["labels"]) == 120
    assert data.train_edges.ndim == 2
    assert data.train_edges.shape[1] == 2


def test_sbm_no_train_test_leakage():
    data = generate_sbm_graph(SBMDGPConfig(n_nodes=100, seed=1))
    train = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in data.train_edges}
    for split in (data.val_split, data.test_split):
        pos = split.edges[split.labels == 1]
        for u, v in pos:
            key = (min(int(u), int(v)), max(int(u), int(v)))
            assert key not in train


def test_sbm_adjacency_from_train_only():
    data = generate_sbm_graph(SBMDGPConfig(n_nodes=80, seed=2))
    adj = data.adjacency
    for u, v in data.train_edges:
        assert adj[int(u), int(v)] == 1.0
        assert adj[int(v), int(u)] == 1.0


def test_temporal_snapshot_ordering():
    data = generate_temporal_graph(TemporalDGPConfig(n_nodes=60, n_snapshots=8, seed=3))
    assert data.n_snapshots == 8
    assert len(data.ground_truth["latent_trajectory"]) == 9
    assert data.forecast_split.edges.ndim == 2
    assert len(data.forecast_split.labels) == data.forecast_split.n_edges


def test_temporal_symmetric_snapshots():
    data = generate_temporal_graph(TemporalDGPConfig(n_nodes=50, n_snapshots=6, seed=4))
    for snap in data.snapshots:
        np.testing.assert_allclose(snap, snap.T)
