"""Rolling-window GCN and temporal GNN architectures for dynamic graphs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from graph_ml.data.base import TemporalGraphDataset
from graph_ml.models.gcn import GCN
from graph_ml.models.layers import normalized_adjacency
from graph_ml.models.link_predictor import dot_product_scores
from graph_ml.temporal.metrics import TemporalMetrics, evaluate_temporal_forecast
from graph_ml.utils.seed import set_torch_seed

logger = logging.getLogger(__name__)


@dataclass
class RollingTrainConfig:
    epochs: int = 60
    lr: float = 0.01
    hidden_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.2
    history_window: int = 3
    neg_ratio: float = 1.0
    seed: int = 42


@dataclass
class RollingGCNResult:
    model_name: str
    train_loss: float
    forecast_metrics: TemporalMetrics


def _cumulative_adjacency(snapshots: list[np.ndarray], end_idx: int) -> np.ndarray:
    """Union adjacency over snapshots [0, end_idx]."""
    adj = np.zeros_like(snapshots[0], dtype=np.float32)
    for t in range(end_idx + 1):
        adj = np.maximum(adj, snapshots[t])
    return adj


def _snapshot_pos_edges(adj: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(adj.shape[0], k=1)
    mask = adj[idx] > 0
    return np.column_stack([idx[0][mask], idx[1][mask]]).astype(np.int64)


def _sample_negatives(
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


def _build_training_pairs(
    dataset: TemporalGraphDataset,
    neg_ratio: float,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Create (adjacency, pos_edges, neg_edges) tuples for each forecast step."""
    pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for t in range(1, dataset.n_snapshots):
        cum_adj = _cumulative_adjacency(dataset.snapshots, t - 1)
        pos = _snapshot_pos_edges(dataset.snapshots[t])
        if len(pos) == 0:
            continue
        n_pos = min(len(pos), 40)
        rng.shuffle(pos)
        pos = pos[:n_pos]
        forbidden = {
            (min(int(u), int(v)), max(int(u), int(v)))
            for snap in dataset.snapshots[: t + 1]
            for u, v in _snapshot_pos_edges(snap)
        }
        neg = _sample_negatives(dataset.n_nodes, len(pos), forbidden, neg_ratio, rng)
        pairs.append((cum_adj, pos, neg))
    return pairs


def fit_rolling_gcn(
    dataset: TemporalGraphDataset,
    config: RollingTrainConfig | None = None,
) -> RollingGCNResult:
    """
    Train a GCN on cumulative snapshots to predict edges at the next timestep.

    For each t, adjacency = union(snapshots[0:t]), target = edges in snapshot t.
    """
    cfg = config or RollingTrainConfig()
    set_torch_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    x = torch.as_tensor(dataset.node_features, dtype=torch.float32)
    encoder = GCN(
        dataset.feature_dim,
        cfg.hidden_dim,
        cfg.hidden_dim,
        cfg.n_layers,
        cfg.dropout,
    )
    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.lr)
    criterion = nn.BCEWithLogitsLoss()

    train_pairs = _build_training_pairs(dataset, cfg.neg_ratio, rng)
    if not train_pairs:
        raise ValueError("No training snapshots available for rolling GCN.")

    encoder.train()
    last_loss = 0.0
    for _ in range(cfg.epochs):
        total_loss = 0.0
        for cum_adj, pos, neg in train_pairs:
            edges = np.vstack([pos, neg])
            labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            adj = torch.as_tensor(cum_adj, dtype=torch.float32)
            edge_t = torch.as_tensor(edges, dtype=torch.long)
            label_t = torch.as_tensor(labels, dtype=torch.float32)

            optimizer.zero_grad()
            emb = encoder(x, adj)
            logits = dot_product_scores(emb, edge_t)
            loss = criterion(logits, label_t)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        last_loss = total_loss / len(train_pairs)

    encoder.eval()
    forecast_adj = _cumulative_adjacency(dataset.snapshots, dataset.n_snapshots - 2)
    adj_t = torch.as_tensor(forecast_adj, dtype=torch.float32)
    edges = dataset.forecast_split.edges
    with torch.no_grad():
        emb = encoder(x, adj_t)
        edge_t = torch.as_tensor(edges, dtype=torch.long)
        scores = dot_product_scores(emb, edge_t).cpu().numpy()

    metrics = evaluate_temporal_forecast(scores, dataset.forecast_split.labels)
    return RollingGCNResult(
        model_name="RollingGCN",
        train_loss=last_loss,
        forecast_metrics=metrics,
    )


# ---------------------------------------------------------------------------
# TemporalGNN: GCN + GRU recurrent update
# ---------------------------------------------------------------------------


class TemporalGNN(nn.Module):
    """Temporal GNN combining spatial GCN with GRU recurrence.

    At each timestep *t* the model computes:
        z_t = GCN(A_t, X_t)
        h_t = GRU(z_t, h_{t-1})

    This lets the network capture both structural patterns within each
    snapshot and evolutionary dynamics across time.

    Args:
        in_dim: Node feature dimensionality.
        hidden_dim: Width of the GCN and GRU hidden state.
        out_dim: Final embedding dimensionality.
        n_gcn_layers: Depth of the per-snapshot GCN encoder.
        dropout: Dropout applied inside the GCN.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        n_gcn_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gcn = GCN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=n_gcn_layers,
            dropout=dropout,
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x_seq: list[torch.Tensor],
        adj_seq: list[torch.Tensor],
        h_0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process a sequence of graph snapshots.

        Args:
            x_seq: List of node feature tensors ``[(N, in_dim), ...]`` per timestep.
            adj_seq: List of adjacency matrices ``[(N, N), ...]`` per timestep.
            h_0: Optional initial hidden state ``(N, hidden_dim)``.

        Returns:
            Final node embeddings ``(N, out_dim)`` after the last timestep.

        Raises:
            ValueError: If ``x_seq`` and ``adj_seq`` have different lengths.
        """
        if len(x_seq) != len(adj_seq):
            raise ValueError("x_seq and adj_seq must have the same length")

        n_nodes = x_seq[0].shape[0]
        device = x_seq[0].device
        h = h_0 if h_0 is not None else torch.zeros(n_nodes, self.hidden_dim, device=device)

        for x_t, adj_t in zip(x_seq, adj_seq):
            z_t = self.gcn(x_t, adj_t)
            h = self.gru(z_t, h)

        return self.out_proj(h)

    def forward_all_steps(
        self,
        x_seq: list[torch.Tensor],
        adj_seq: list[torch.Tensor],
        h_0: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Return embeddings at every timestep.

        Args:
            x_seq: List of node feature tensors per timestep.
            adj_seq: List of adjacency matrices per timestep.
            h_0: Optional initial hidden state.

        Returns:
            List of node embedding tensors ``[(N, out_dim), ...]``.
        """
        if len(x_seq) != len(adj_seq):
            raise ValueError("x_seq and adj_seq must have the same length")

        n_nodes = x_seq[0].shape[0]
        device = x_seq[0].device
        h = h_0 if h_0 is not None else torch.zeros(n_nodes, self.hidden_dim, device=device)
        outputs: list[torch.Tensor] = []

        for x_t, adj_t in zip(x_seq, adj_seq):
            z_t = self.gcn(x_t, adj_t)
            h = self.gru(z_t, h)
            outputs.append(self.out_proj(h))

        return outputs


# ---------------------------------------------------------------------------
# TGN_Module: Simplified Temporal Graph Network
# ---------------------------------------------------------------------------


@dataclass
class TGNConfig:
    """Configuration for :class:`TGN_Module`.

    Args:
        node_dim: Raw node feature size.
        edge_dim: Raw edge feature size.
        memory_dim: Dimensionality of node memory vectors.
        time_dim: Dimensionality of time encoding.
        hidden_dim: Width of the message and memory update MLPs.
    """

    node_dim: int = 16
    edge_dim: int = 8
    memory_dim: int = 64
    time_dim: int = 16
    hidden_dim: int = 64


class TGN_Module(nn.Module):
    """Simplified Temporal Graph Network (Rossi et al., 2020).

    Maintains per-node memory vectors updated by an interaction message
    function and a GRU-based memory updater:
        1. **Message function**: m_i(t) = MLP(s_i(t^-) || s_j(t^-) || e_{ij}(t) || φ(t))
        2. **Memory update**: s_i(t) = GRU(m_i(t), s_i(t^-))

    where φ(t) is a learnable time encoding.

    Args:
        n_nodes: Total number of nodes (for memory allocation).
        config: :class:`TGNConfig` with model hyperparameters.
    """

    def __init__(self, n_nodes: int, config: TGNConfig | None = None) -> None:
        super().__init__()
        cfg = config or TGNConfig()
        self.n_nodes = n_nodes
        self.memory_dim = cfg.memory_dim

        self.memory = nn.Parameter(
            torch.zeros(n_nodes, cfg.memory_dim), requires_grad=False
        )
        self.last_update = nn.Parameter(
            torch.zeros(n_nodes), requires_grad=False
        )

        msg_in = cfg.memory_dim * 2 + cfg.edge_dim + cfg.time_dim
        self.message_fn = nn.Sequential(
            nn.Linear(msg_in, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.memory_dim),
        )
        self.memory_updater = nn.GRUCell(cfg.memory_dim, cfg.memory_dim)

        self.time_enc = nn.Linear(1, cfg.time_dim)

    def _encode_time(self, dt: torch.Tensor) -> torch.Tensor:
        """Encode time deltas with a learned linear projection.

        Args:
            dt: Time deltas ``(B,)``.

        Returns:
            Time features ``(B, time_dim)``.
        """
        return torch.cos(self.time_enc(dt.unsqueeze(-1)))

    def compute_messages(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        t: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute interaction messages for a batch of events.

        Args:
            src: Source node indices ``(B,)``.
            dst: Destination node indices ``(B,)``.
            t: Event timestamps ``(B,)``.
            edge_feat: Edge features ``(B, edge_dim)``.

        Returns:
            Messages ``(B, memory_dim)``.
        """
        src_mem = self.memory[src].detach()
        dst_mem = self.memory[dst].detach()
        dt = t - self.last_update[src].detach()
        time_feat = self._encode_time(dt)
        inp = torch.cat([src_mem, dst_mem, edge_feat, time_feat], dim=-1)
        return self.message_fn(inp)

    def update_memory(
        self,
        node_ids: torch.Tensor,
        messages: torch.Tensor,
        t: torch.Tensor,
    ) -> None:
        """Update memory for a set of nodes given aggregated messages.

        Args:
            node_ids: Node indices to update ``(B,)``.
            messages: Aggregated messages ``(B, memory_dim)``.
            t: Timestamps of the triggering events ``(B,)``.
        """
        current = self.memory[node_ids].detach()
        updated = self.memory_updater(messages, current)
        self.memory.data[node_ids] = updated.detach()
        self.last_update.data[node_ids] = t.detach()

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        t: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Process a batch of temporal edges: compute messages and update memory.

        Args:
            src: Source node indices ``(B,)``.
            dst: Destination node indices ``(B,)``.
            t: Timestamps ``(B,)``.
            edge_feat: Edge features ``(B, edge_dim)``.

        Returns:
            Updated source node memory embeddings ``(B, memory_dim)``.
        """
        msgs = self.compute_messages(src, dst, t, edge_feat)
        self.update_memory(src, msgs, t)
        self.update_memory(dst, msgs, t)
        return self.memory[src]

    def reset_memory(self) -> None:
        """Zero-out all node memories and timestamps."""
        self.memory.data.zero_()
        self.last_update.data.zero_()

    def get_memory(self, node_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Retrieve current memory vectors.

        Args:
            node_ids: Optional node indices. If ``None``, return all memories.

        Returns:
            Memory tensor ``(B, memory_dim)`` or ``(N, memory_dim)``.
        """
        if node_ids is None:
            return self.memory.detach()
        return self.memory[node_ids].detach()


# ---------------------------------------------------------------------------
# TemporalEdgeDataset: PyTorch Dataset for temporal interaction data
# ---------------------------------------------------------------------------


class TemporalEdgeDataset(Dataset):
    """Dataset of temporal edges with negative sampling.

    Stores events as ``(src, dst, timestamp, edge_features)`` tuples and
    provides on-the-fly uniform negative sampling.

    Args:
        src: Source node indices ``(E,)``.
        dst: Destination node indices ``(E,)``.
        timestamps: Event timestamps ``(E,)``.
        edge_features: Edge feature matrix ``(E, edge_dim)``.
        n_nodes: Total number of nodes (for negative sampling bounds).
        neg_ratio: Number of negatives per positive edge.
        seed: Random seed for reproducible negative sampling.
    """

    def __init__(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        timestamps: np.ndarray,
        edge_features: np.ndarray,
        n_nodes: int,
        neg_ratio: int = 1,
        seed: int = 42,
    ) -> None:
        if not (len(src) == len(dst) == len(timestamps) == len(edge_features)):
            raise ValueError("All input arrays must have the same length")

        sort_idx = np.argsort(timestamps)
        self.src = torch.as_tensor(src[sort_idx], dtype=torch.long)
        self.dst = torch.as_tensor(dst[sort_idx], dtype=torch.long)
        self.timestamps = torch.as_tensor(timestamps[sort_idx], dtype=torch.float32)
        self.edge_features = torch.as_tensor(edge_features[sort_idx], dtype=torch.float32)
        self.n_nodes = n_nodes
        self.neg_ratio = neg_ratio
        self._rng = np.random.default_rng(seed)

        self._positive_set: set[tuple[int, int]] = set()
        for s, d in zip(src, dst):
            self._positive_set.add((int(s), int(d)))

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single temporal edge with negative samples.

        Args:
            idx: Index into the sorted edge list.

        Returns:
            Dict with keys ``src``, ``dst``, ``t``, ``feat``, ``neg_dst``
            where ``neg_dst`` has shape ``(neg_ratio,)``.
        """
        neg_dsts: list[int] = []
        attempts = 0
        s = int(self.src[idx].item())
        while len(neg_dsts) < self.neg_ratio and attempts < self.neg_ratio * 50:
            cand = int(self._rng.integers(0, self.n_nodes))
            if cand != s and (s, cand) not in self._positive_set:
                neg_dsts.append(cand)
            attempts += 1

        while len(neg_dsts) < self.neg_ratio:
            neg_dsts.append(int(self._rng.integers(0, self.n_nodes)))

        return {
            "src": self.src[idx],
            "dst": self.dst[idx],
            "t": self.timestamps[idx],
            "feat": self.edge_features[idx],
            "neg_dst": torch.tensor(neg_dsts, dtype=torch.long),
        }

    def temporal_split(
        self, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> tuple[TemporalEdgeDataset, TemporalEdgeDataset, TemporalEdgeDataset]:
        """Split edges chronologically into train / val / test.

        Args:
            train_ratio: Fraction of edges for training.
            val_ratio: Fraction of edges for validation.

        Returns:
            Tuple of ``(train_dataset, val_dataset, test_dataset)``.
        """
        n = len(self)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits: list[TemporalEdgeDataset] = []
        for start, end in [
            (0, n_train),
            (n_train, n_train + n_val),
            (n_train + n_val, n),
        ]:
            s = self.src[start:end].numpy()
            d = self.dst[start:end].numpy()
            t = self.timestamps[start:end].numpy()
            f = self.edge_features[start:end].numpy()
            splits.append(
                TemporalEdgeDataset(
                    src=s, dst=d, timestamps=t, edge_features=f,
                    n_nodes=self.n_nodes, neg_ratio=self.neg_ratio,
                    seed=int(self._rng.integers(0, 2**31)),
                )
            )
        return splits[0], splits[1], splits[2]
