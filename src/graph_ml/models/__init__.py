"""Graph neural network models."""

from graph_ml.models.gcn import GCN
from graph_ml.models.graphsage import GraphSAGE
from graph_ml.models.link_predictor import dot_product_scores, mlp_link_scores

__all__ = ["GCN", "GraphSAGE", "dot_product_scores", "mlp_link_scores"]
