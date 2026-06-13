"""Synthetic graph data generators."""

from graph_ml.data.base import GraphDataset, TemporalGraphDataset
from graph_ml.data.static_graph_dgp import SBMDGPConfig, generate_sbm_graph
from graph_ml.data.temporal_graph_dgp import TemporalDGPConfig, generate_temporal_graph

__all__ = [
    "GraphDataset",
    "TemporalGraphDataset",
    "SBMDGPConfig",
    "generate_sbm_graph",
    "TemporalDGPConfig",
    "generate_temporal_graph",
]
