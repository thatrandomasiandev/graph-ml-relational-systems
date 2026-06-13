"""Synthetic graph data generators and OGB loaders."""

from graph_ml.data.base import GraphDataset, TemporalGraphDataset
from graph_ml.data.ogb_loader import OGBDataLoader, OGBDataLoaderConfig, load_ogbl_collab, load_ogbn_arxiv
from graph_ml.data.static_graph_dgp import SBMDGPConfig, generate_sbm_graph
from graph_ml.data.temporal_graph_dgp import TemporalDGPConfig, generate_temporal_graph

__all__ = [
    "GraphDataset",
    "TemporalGraphDataset",
    "OGBDataLoader",
    "OGBDataLoaderConfig",
    "SBMDGPConfig",
    "generate_sbm_graph",
    "TemporalDGPConfig",
    "generate_temporal_graph",
    "load_ogbl_collab",
    "load_ogbn_arxiv",
]
