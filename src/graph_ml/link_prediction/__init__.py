"""Link prediction training and metrics."""

from graph_ml.link_prediction.metrics import LinkPredictionMetrics, evaluate_link_prediction
from graph_ml.link_prediction.trainer import LinkPredictionResult, TrainConfig, fit_link_predictor

__all__ = [
    "LinkPredictionMetrics",
    "evaluate_link_prediction",
    "LinkPredictionResult",
    "TrainConfig",
    "fit_link_predictor",
]
