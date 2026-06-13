"""OGB dataset loader with automatic fallback to synthetic data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from graph_ml.data.base import EdgeSplit, GraphDataset
from graph_ml.utils.seed import set_seed

logger = logging.getLogger(__name__)

_OGB_AVAILABLE: bool
try:
    from ogb.nodeproppred import NodePropPredDataset  # type: ignore[import-untyped]
    from ogb.linkproppred import LinkPropPredDataset  # type: ignore[import-untyped]

    _OGB_AVAILABLE = True
except ImportError:
    _OGB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers for synthetic fallback
# ---------------------------------------------------------------------------


def _synthetic_node_classification_graph(
    n_nodes: int = 500,
    feature_dim: int = 128,
    n_classes: int = 40,
    edge_density: float = 0.01,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate a synthetic attributed graph resembling ogbn-arxiv.

    Args:
        n_nodes: Number of nodes.
        feature_dim: Node feature dimensionality.
        n_classes: Number of node classes.
        edge_density: Approximate fraction of possible edges present.
        seed: Random seed.

    Returns:
        Dict with keys ``node_features``, ``labels``, ``edges``,
        ``train_idx``, ``val_idx``, ``test_idx``.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n_nodes)
    centroids = rng.standard_normal((n_classes, feature_dim)).astype(np.float32)
    features = centroids[labels] + 0.3 * rng.standard_normal(
        (n_nodes, feature_dim)
    ).astype(np.float32)

    n_edges = max(10, int(n_nodes * (n_nodes - 1) / 2 * edge_density))
    src = rng.integers(0, n_nodes, size=n_edges)
    dst = rng.integers(0, n_nodes, size=n_edges)
    mask = src != dst
    edges = np.column_stack([src[mask], dst[mask]]).astype(np.int64)

    idx = rng.permutation(n_nodes)
    n_train = int(0.7 * n_nodes)
    n_val = int(0.15 * n_nodes)

    return {
        "node_features": features,
        "labels": labels,
        "edges": edges,
        "train_idx": idx[:n_train],
        "val_idx": idx[n_train : n_train + n_val],
        "test_idx": idx[n_train + n_val :],
    }


def _synthetic_link_prediction_graph(
    n_nodes: int = 300,
    feature_dim: int = 64,
    edge_density: float = 0.02,
    seed: int = 42,
) -> GraphDataset:
    """Generate a synthetic link prediction graph resembling ogbl-collab.

    Args:
        n_nodes: Number of nodes.
        feature_dim: Node feature dimensionality.
        edge_density: Approximate fraction of possible edges present.
        seed: Random seed.

    Returns:
        :class:`GraphDataset` with train/val/test splits.
    """
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n_nodes, feature_dim)).astype(np.float32)

    n_edges = max(20, int(n_nodes * (n_nodes - 1) / 2 * edge_density))
    src = rng.integers(0, n_nodes, size=n_edges * 2)
    dst = rng.integers(0, n_nodes, size=n_edges * 2)
    mask = src != dst
    pairs = np.column_stack([src[mask], dst[mask]])

    unique: set[tuple[int, int]] = set()
    deduped: list[tuple[int, int]] = []
    for u, v in pairs:
        key = (min(int(u), int(v)), max(int(u), int(v)))
        if key not in unique:
            unique.add(key)
            deduped.append(key)
    edges = np.array(deduped, dtype=np.int64)
    rng.shuffle(edges)

    n_total = len(edges)
    n_train = max(1, int(0.7 * n_total))
    n_val = max(1, int(0.15 * n_total))

    train_edges = edges[:n_train]
    val_pos = edges[n_train : n_train + n_val]
    test_pos = edges[n_train + n_val :]

    all_edges_set = set(map(tuple, edges.tolist()))

    def _neg_sample(n: int) -> np.ndarray:
        negs: list[tuple[int, int]] = []
        attempts = 0
        while len(negs) < n and attempts < n * 100:
            u_r = int(rng.integers(0, n_nodes))
            v_r = int(rng.integers(0, n_nodes))
            if u_r != v_r and (min(u_r, v_r), max(u_r, v_r)) not in all_edges_set:
                negs.append((min(u_r, v_r), max(u_r, v_r)))
            attempts += 1
        return np.array(negs, dtype=np.int64) if negs else np.empty((0, 2), dtype=np.int64)

    val_neg = _neg_sample(len(val_pos))
    test_neg = _neg_sample(len(test_pos))

    val_split = EdgeSplit(
        edges=np.vstack([val_pos, val_neg]) if len(val_neg) > 0 else val_pos,
        labels=np.concatenate([np.ones(len(val_pos)), np.zeros(len(val_neg))]),
    )
    test_split = EdgeSplit(
        edges=np.vstack([test_pos, test_neg]) if len(test_neg) > 0 else test_pos,
        labels=np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))]),
    )

    return GraphDataset(
        node_features=features,
        train_edges=train_edges,
        val_split=val_split,
        test_split=test_split,
        metadata={"source": "synthetic_collab_fallback", "n_nodes": n_nodes},
    )


# ---------------------------------------------------------------------------
# OGBDataLoader
# ---------------------------------------------------------------------------


@dataclass
class OGBDataLoaderConfig:
    """Configuration for :class:`OGBDataLoader`.

    Args:
        data_dir: Root directory for OGB downloads.
        seed: Random seed used for synthetic fallback generation.
        force_synthetic: If ``True``, always use synthetic data even when OGB
            is available.
    """

    data_dir: str = "./data/ogb"
    seed: int = 42
    force_synthetic: bool = False


class OGBDataLoader:
    """Thin loader wrapping OGB datasets with automatic synthetic fallback.

    When the ``ogb`` package is installed, loads the requested dataset from
    OGB. Otherwise (or when ``force_synthetic=True``), generates a synthetic
    graph with similar characteristics.

    Args:
        config: :class:`OGBDataLoaderConfig` options.
    """

    def __init__(self, config: OGBDataLoaderConfig | None = None) -> None:
        self.config = config or OGBDataLoaderConfig()
        if _OGB_AVAILABLE and not self.config.force_synthetic:
            logger.info("OGB package available — will attempt real datasets")
        else:
            logger.info("OGB package unavailable or force_synthetic=True — using fallback")

    @property
    def ogb_available(self) -> bool:
        """Whether the OGB package can be imported."""
        return _OGB_AVAILABLE and not self.config.force_synthetic

    def load_node_dataset(self, name: str) -> dict[str, Any]:
        """Load an OGB node-property-prediction dataset.

        Args:
            name: OGB dataset name, e.g. ``"ogbn-arxiv"``.

        Returns:
            Dict with ``node_features``, ``labels``, ``edges``, and split indices.
        """
        if self.ogb_available:
            try:
                dataset = NodePropPredDataset(name=name, root=self.config.data_dir)
                graph, labels = dataset[0]
                split_idx = dataset.get_idx_split()
                return {
                    "node_features": graph["node_feat"],
                    "labels": labels.squeeze(),
                    "edges": graph["edge_index"].T,
                    "train_idx": split_idx["train"],
                    "val_idx": split_idx["valid"],
                    "test_idx": split_idx["test"],
                }
            except Exception:
                logger.warning("Failed to load %s from OGB, falling back to synthetic", name)

        return _synthetic_node_classification_graph(seed=self.config.seed)

    def load_link_dataset(self, name: str) -> GraphDataset:
        """Load an OGB link-property-prediction dataset.

        Args:
            name: OGB dataset name, e.g. ``"ogbl-collab"``.

        Returns:
            :class:`GraphDataset` with train/val/test link splits.
        """
        if self.ogb_available:
            try:
                dataset = LinkPropPredDataset(name=name, root=self.config.data_dir)
                graph = dataset[0]
                split_edge = dataset.get_edge_split()

                n_nodes = int(graph["num_nodes"])
                feat_dim = 64
                rng = np.random.default_rng(self.config.seed)
                features = rng.standard_normal((n_nodes, feat_dim)).astype(np.float32)

                train_edges = split_edge["train"]["edge"].numpy()
                val_pos = split_edge["valid"]["edge"].numpy()
                val_neg = split_edge["valid"]["edge_neg"].numpy()
                test_pos = split_edge["test"]["edge"].numpy()
                test_neg = split_edge["test"]["edge_neg"].numpy()

                if val_neg.ndim > 2:
                    val_neg = val_neg[:, 0, :]
                if test_neg.ndim > 2:
                    test_neg = test_neg[:, 0, :]

                val_split = EdgeSplit(
                    edges=np.vstack([val_pos, val_neg]),
                    labels=np.concatenate([np.ones(len(val_pos)), np.zeros(len(val_neg))]),
                )
                test_split = EdgeSplit(
                    edges=np.vstack([test_pos, test_neg]),
                    labels=np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))]),
                )

                return GraphDataset(
                    node_features=features,
                    train_edges=train_edges,
                    val_split=val_split,
                    test_split=test_split,
                    metadata={"source": name, "n_nodes": n_nodes},
                )
            except Exception:
                logger.warning("Failed to load %s from OGB, falling back to synthetic", name)

        return _synthetic_link_prediction_graph(seed=self.config.seed)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def load_ogbn_arxiv(
    data_dir: str = "./data/ogb",
    seed: int = 42,
    force_synthetic: bool = False,
) -> dict[str, Any]:
    """Load ogbn-arxiv with fallback to synthetic node classification data.

    Args:
        data_dir: Download root for OGB.
        seed: Random seed for synthetic fallback.
        force_synthetic: Skip OGB even if available.

    Returns:
        Dict with ``node_features``, ``labels``, ``edges``, and split indices.
    """
    loader = OGBDataLoader(OGBDataLoaderConfig(
        data_dir=data_dir, seed=seed, force_synthetic=force_synthetic,
    ))
    return loader.load_node_dataset("ogbn-arxiv")


def load_ogbl_collab(
    data_dir: str = "./data/ogb",
    seed: int = 42,
    force_synthetic: bool = False,
) -> GraphDataset:
    """Load ogbl-collab with fallback to synthetic link prediction data.

    Args:
        data_dir: Download root for OGB.
        seed: Random seed for synthetic fallback.
        force_synthetic: Skip OGB even if available.

    Returns:
        :class:`GraphDataset` with train/val/test link splits.
    """
    loader = OGBDataLoader(OGBDataLoaderConfig(
        data_dir=data_dir, seed=seed, force_synthetic=force_synthetic,
    ))
    return loader.load_link_dataset("ogbl-collab")
