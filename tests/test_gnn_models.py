"""Tests for GCN, GAT, GIN, and TemporalGNN models."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from graph_ml.models.gcn import GCN, GraphAttentionNetwork, GraphIsomorphismNetwork
from graph_ml.models.graphsage import GraphSAGE, InductiveGraphSAGE, SAGE_AGGREGATORS
from graph_ml.models.pooling import (
    DiffPool,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from graph_ml.temporal.rolling_gnn import TemporalGNN, TGN_Module, TGNConfig, TemporalEdgeDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 42


@pytest.fixture()
def simple_graph() -> tuple[torch.Tensor, torch.Tensor]:
    """5-node graph with 16-dim features and a small adjacency."""
    torch.manual_seed(SEED)
    x = torch.randn(5, 16)
    adj = torch.zeros(5, 5)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return x, adj


@pytest.fixture()
def batched_graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two small graphs batched via block-diagonal adjacency."""
    torch.manual_seed(SEED)
    x1 = torch.randn(3, 8)
    x2 = torch.randn(4, 8)
    adj1 = torch.ones(3, 3) - torch.eye(3)
    adj2 = torch.ones(4, 4) - torch.eye(4)
    x = torch.cat([x1, x2], dim=0)
    adj = torch.zeros(7, 7)
    adj[:3, :3] = adj1
    adj[3:, 3:] = adj2
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    return x, adj, batch


# ---------------------------------------------------------------------------
# GCN tests
# ---------------------------------------------------------------------------


class TestGCN:
    def test_output_shape(self, simple_graph: tuple[torch.Tensor, torch.Tensor]) -> None:
        x, adj = simple_graph
        model = GCN(in_dim=16, hidden_dim=32, out_dim=8, n_layers=2)
        out = model(x, adj)
        assert out.shape == (5, 8)

    def test_single_layer(self, simple_graph: tuple[torch.Tensor, torch.Tensor]) -> None:
        x, adj = simple_graph
        model = GCN(in_dim=16, hidden_dim=32, out_dim=8, n_layers=1)
        out = model(x, adj)
        assert out.shape == (5, 8)

    def test_deep_network(self, simple_graph: tuple[torch.Tensor, torch.Tensor]) -> None:
        x, adj = simple_graph
        model = GCN(in_dim=16, hidden_dim=32, out_dim=8, n_layers=5)
        out = model(x, adj)
        assert out.shape == (5, 8)
        assert torch.isfinite(out).all()

    def test_invalid_layers(self) -> None:
        with pytest.raises(ValueError, match="n_layers must be >= 1"):
            GCN(in_dim=16, n_layers=0)


# ---------------------------------------------------------------------------
# GAT tests
# ---------------------------------------------------------------------------


class TestGAT:
    def test_output_shape(self, simple_graph: tuple[torch.Tensor, torch.Tensor]) -> None:
        x, adj = simple_graph
        model = GraphAttentionNetwork(
            in_dim=16, hidden_dim=8, out_dim=32, n_heads=4
        )
        out = model(x, adj)
        assert out.shape == (5, 32)

    def test_attention_weights_sum_to_one(
        self, simple_graph: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Attention coefficients for each node must sum to 1.0 across neighbors."""
        x, adj = simple_graph
        model = GraphAttentionNetwork(
            in_dim=16, hidden_dim=8, out_dim=32, n_heads=4, dropout=0.0
        )
        model.eval()
        with torch.no_grad():
            _ = model(x, adj)

        attn = model.last_attention
        assert attn is not None
        assert attn.shape == (4, 5, 5)

        row_sums = attn.sum(dim=-1)
        torch.testing.assert_close(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_single_head(self, simple_graph: tuple[torch.Tensor, torch.Tensor]) -> None:
        x, adj = simple_graph
        model = GraphAttentionNetwork(
            in_dim=16, hidden_dim=8, out_dim=16, n_heads=1
        )
        out = model(x, adj)
        assert out.shape == (5, 16)

    def test_gradient_flow(self, simple_graph: tuple[torch.Tensor, torch.Tensor]) -> None:
        x, adj = simple_graph
        model = GraphAttentionNetwork(in_dim=16, hidden_dim=8, out_dim=4, n_heads=2)
        out = model(x, adj)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ---------------------------------------------------------------------------
# GIN tests
# ---------------------------------------------------------------------------


class TestGIN:
    def test_output_shape(self, simple_graph: tuple[torch.Tensor, torch.Tensor]) -> None:
        x, adj = simple_graph
        model = GraphIsomorphismNetwork(
            in_dim=16, hidden_dim=32, out_dim=8, n_layers=2
        )
        out = model(x, adj)
        assert out.shape == (5, 8)

    def test_graph_forward_shape(
        self, batched_graph: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        x, adj, batch = batched_graph
        n_layers = 2
        hidden_dim = 16
        out_dim = 8
        model = GraphIsomorphismNetwork(
            in_dim=8, hidden_dim=hidden_dim, out_dim=out_dim, n_layers=n_layers
        )
        graph_emb = model.graph_forward(x, adj, batch)
        # readout dimensions: initial_proj(out_dim) + layer_0(hidden_dim) + layer_1(out_dim)
        expected_dim = out_dim + hidden_dim * (n_layers - 1) + out_dim
        assert graph_emb.shape == (2, expected_dim)

    def test_distinguishing_capability(self) -> None:
        """GIN should produce different node embeddings for structurally different positions.

        In a line graph (0-1-2-3), the center nodes (degree 2) should get
        different embeddings from the endpoints (degree 1). This validates
        that GIN captures structural differences via injective aggregation.
        """
        torch.manual_seed(SEED)

        x_same = torch.ones(4, 8)
        adj_line = torch.zeros(4, 4)
        adj_line[0, 1] = adj_line[1, 0] = 1
        adj_line[1, 2] = adj_line[2, 1] = 1
        adj_line[2, 3] = adj_line[3, 2] = 1

        model = GraphIsomorphismNetwork(
            in_dim=8, hidden_dim=32, out_dim=16, n_layers=3
        )
        model.eval()

        with torch.no_grad():
            emb = model(x_same, adj_line)

        endpoint_emb = emb[0]  # degree-1 node
        center_emb = emb[1]    # degree-2 node
        diff = (endpoint_emb - center_emb).norm().item()
        assert diff > 1e-4, (
            f"GIN failed to distinguish degree-1 vs degree-2 nodes "
            f"(L2 diff={diff:.6f})"
        )

    def test_single_layer(self, simple_graph: tuple[torch.Tensor, torch.Tensor]) -> None:
        x, adj = simple_graph
        model = GraphIsomorphismNetwork(in_dim=16, hidden_dim=32, out_dim=8, n_layers=1)
        out = model(x, adj)
        assert out.shape == (5, 8)


# ---------------------------------------------------------------------------
# GraphSAGE tests
# ---------------------------------------------------------------------------


class TestGraphSAGE:
    def test_transductive_shape(
        self, simple_graph: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        x, adj = simple_graph
        model = GraphSAGE(in_dim=16, hidden_dim=32, out_dim=8)
        assert model(x, adj).shape == (5, 8)

    @pytest.mark.parametrize("agg", ["mean", "max"])
    def test_inductive_aggregators(
        self, simple_graph: tuple[torch.Tensor, torch.Tensor], agg: str
    ) -> None:
        x, adj = simple_graph
        model = InductiveGraphSAGE(
            in_dim=16, hidden_dim=32, out_dim=8, aggregator=agg
        )
        out = model(x, adj)
        assert out.shape == (5, 8)
        assert torch.isfinite(out).all()

    def test_inductive_unseen_nodes(self) -> None:
        """InductiveGraphSAGE should handle graphs with more nodes than training."""
        torch.manual_seed(SEED)
        model = InductiveGraphSAGE(in_dim=4, hidden_dim=8, out_dim=4)
        x_train = torch.randn(3, 4)
        adj_train = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float)
        _ = model(x_train, adj_train)

        x_new = torch.randn(5, 4)
        adj_new = torch.zeros(5, 5)
        adj_new[0, 1] = adj_new[1, 0] = 1
        adj_new[2, 3] = adj_new[3, 2] = 1
        out = model(x_new, adj_new)
        assert out.shape == (5, 4)

    def test_invalid_aggregator(self) -> None:
        with pytest.raises(ValueError, match="Unknown aggregator"):
            InductiveGraphSAGE(in_dim=4, hidden_dim=8, out_dim=4, aggregator="invalid")

    def test_aggregator_registry(self) -> None:
        assert set(SAGE_AGGREGATORS.keys()) == {"mean", "max", "lstm"}


# ---------------------------------------------------------------------------
# TemporalGNN tests
# ---------------------------------------------------------------------------


class TestTemporalGNN:
    def test_output_shape(self) -> None:
        torch.manual_seed(SEED)
        n_nodes, feat_dim, out_dim = 5, 8, 4
        model = TemporalGNN(in_dim=feat_dim, hidden_dim=16, out_dim=out_dim)

        x_seq = [torch.randn(n_nodes, feat_dim) for _ in range(3)]
        adj_seq = [torch.eye(n_nodes) for _ in range(3)]
        out = model(x_seq, adj_seq)
        assert out.shape == (n_nodes, out_dim)

    def test_variable_sequence_lengths(self) -> None:
        """TemporalGNN should handle sequences of different lengths."""
        torch.manual_seed(SEED)
        n_nodes, feat_dim = 4, 8
        model = TemporalGNN(in_dim=feat_dim, hidden_dim=16, out_dim=4)

        for seq_len in [1, 3, 7]:
            x_seq = [torch.randn(n_nodes, feat_dim) for _ in range(seq_len)]
            adj_seq = [torch.eye(n_nodes) for _ in range(seq_len)]
            out = model(x_seq, adj_seq)
            assert out.shape == (n_nodes, 4), f"Failed for seq_len={seq_len}"

    def test_all_steps_output(self) -> None:
        torch.manual_seed(SEED)
        n_nodes, feat_dim, n_steps = 5, 8, 4
        model = TemporalGNN(in_dim=feat_dim, hidden_dim=16, out_dim=4)

        x_seq = [torch.randn(n_nodes, feat_dim) for _ in range(n_steps)]
        adj_seq = [torch.eye(n_nodes) for _ in range(n_steps)]
        all_out = model.forward_all_steps(x_seq, adj_seq)
        assert len(all_out) == n_steps
        for o in all_out:
            assert o.shape == (n_nodes, 4)

    def test_mismatched_sequences_raise(self) -> None:
        model = TemporalGNN(in_dim=8, hidden_dim=16, out_dim=4)
        x_seq = [torch.randn(5, 8) for _ in range(3)]
        adj_seq = [torch.eye(5) for _ in range(2)]
        with pytest.raises(ValueError, match="same length"):
            model(x_seq, adj_seq)

    def test_gradient_flow(self) -> None:
        torch.manual_seed(SEED)
        model = TemporalGNN(in_dim=8, hidden_dim=16, out_dim=4)
        x_seq = [torch.randn(4, 8) for _ in range(3)]
        adj_seq = [torch.eye(4) for _ in range(3)]
        out = model(x_seq, adj_seq)
        out.sum().backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ---------------------------------------------------------------------------
# TGN_Module tests
# ---------------------------------------------------------------------------


class TestTGNModule:
    def test_forward_updates_memory(self) -> None:
        torch.manual_seed(SEED)
        n_nodes = 10
        config = TGNConfig(edge_dim=4, memory_dim=16, time_dim=8, hidden_dim=16)
        module = TGN_Module(n_nodes=n_nodes, config=config)

        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([3, 4, 5])
        t = torch.tensor([1.0, 2.0, 3.0])
        feat = torch.randn(3, 4)

        mem_before = module.get_memory().clone()
        module(src, dst, t, feat)
        mem_after = module.get_memory()

        changed_nodes = torch.tensor([0, 1, 2, 3, 4, 5])
        assert not torch.allclose(mem_before[changed_nodes], mem_after[changed_nodes])

    def test_reset_memory(self) -> None:
        n_nodes = 5
        module = TGN_Module(n_nodes=n_nodes)
        module.memory.data.fill_(1.0)
        module.reset_memory()
        assert (module.memory == 0).all()


# ---------------------------------------------------------------------------
# TemporalEdgeDataset tests
# ---------------------------------------------------------------------------


class TestTemporalEdgeDataset:
    def test_basic_access(self) -> None:
        rng = np.random.default_rng(SEED)
        n_edges = 50
        ds = TemporalEdgeDataset(
            src=rng.integers(0, 20, n_edges),
            dst=rng.integers(0, 20, n_edges),
            timestamps=np.sort(rng.uniform(0, 100, n_edges)),
            edge_features=rng.standard_normal((n_edges, 4)).astype(np.float32),
            n_nodes=20,
            neg_ratio=2,
        )
        assert len(ds) == n_edges
        item = ds[0]
        assert item["neg_dst"].shape == (2,)

    def test_temporal_split(self) -> None:
        rng = np.random.default_rng(SEED)
        n = 100
        ds = TemporalEdgeDataset(
            src=rng.integers(0, 30, n),
            dst=rng.integers(0, 30, n),
            timestamps=np.sort(rng.uniform(0, 100, n)),
            edge_features=rng.standard_normal((n, 4)).astype(np.float32),
            n_nodes=30,
        )
        train, val, test = ds.temporal_split(0.7, 0.15)
        assert len(train) + len(val) + len(test) == n


# ---------------------------------------------------------------------------
# Pooling tests
# ---------------------------------------------------------------------------


class TestPooling:
    def test_global_mean_pool(self) -> None:
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out = global_mean_pool(x)
        expected = torch.tensor([[3.0, 4.0]])
        torch.testing.assert_close(out, expected)

    def test_global_max_pool(self) -> None:
        x = torch.tensor([[1.0, 6.0], [3.0, 4.0], [5.0, 2.0]])
        out = global_max_pool(x)
        expected = torch.tensor([[5.0, 6.0]])
        torch.testing.assert_close(out, expected)

    def test_global_add_pool(self) -> None:
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = global_add_pool(x)
        expected = torch.tensor([[4.0, 6.0]])
        torch.testing.assert_close(out, expected)

    def test_global_pools_batched(self) -> None:
        x = torch.tensor([[1.0], [2.0], [10.0], [20.0]])
        batch = torch.tensor([0, 0, 1, 1])
        assert global_mean_pool(x, batch).shape == (2, 1)
        assert global_max_pool(x, batch).shape == (2, 1)
        assert global_add_pool(x, batch).shape == (2, 1)

        mean_out = global_mean_pool(x, batch)
        torch.testing.assert_close(mean_out, torch.tensor([[1.5], [15.0]]))

    def test_diffpool_output_shapes(self) -> None:
        torch.manual_seed(SEED)
        x = torch.randn(10, 8)
        adj = torch.zeros(10, 10)
        for i in range(9):
            adj[i, i + 1] = adj[i + 1, i] = 1.0

        model = DiffPool(in_dim=8, hidden_dim=16, out_dim=8, n_clusters=3)
        x_pool, adj_pool, lp_loss, e_loss = model(x, adj)

        assert x_pool.shape == (3, 8)
        assert adj_pool.shape == (3, 3)
        assert lp_loss.ndim == 0
        assert e_loss.ndim == 0

    def test_diffpool_losses_are_finite(self) -> None:
        torch.manual_seed(SEED)
        x = torch.randn(6, 4)
        adj = torch.ones(6, 6) - torch.eye(6)
        model = DiffPool(in_dim=4, hidden_dim=8, out_dim=4, n_clusters=2)
        _, _, lp_loss, e_loss = model(x, adj)
        assert torch.isfinite(lp_loss)
        assert torch.isfinite(e_loss)
