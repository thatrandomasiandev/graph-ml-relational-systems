"""Graph neural network models."""

from graph_ml.models.gcn import GCN, GraphAttentionNetwork, GraphIsomorphismNetwork
from graph_ml.models.graphsage import GraphSAGE, InductiveGraphSAGE, SAGE_AGGREGATORS
from graph_ml.models.link_predictor import dot_product_scores, mlp_link_scores
from graph_ml.models.pooling import DiffPool, global_add_pool, global_max_pool, global_mean_pool

__all__ = [
    "GCN",
    "GraphAttentionNetwork",
    "GraphIsomorphismNetwork",
    "GraphSAGE",
    "InductiveGraphSAGE",
    "SAGE_AGGREGATORS",
    "DiffPool",
    "dot_product_scores",
    "global_add_pool",
    "global_max_pool",
    "global_mean_pool",
    "mlp_link_scores",
]
