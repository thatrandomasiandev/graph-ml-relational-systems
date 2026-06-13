"""Benchmark runner for link prediction and temporal graph modules."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from graph_ml.data.static_graph_dgp import SBMDGPConfig, generate_sbm_graph
from graph_ml.data.temporal_graph_dgp import TemporalDGPConfig, generate_temporal_graph
from graph_ml.link_prediction.trainer import TrainConfig, fit_link_predictor
from graph_ml.temporal.rolling_gnn import RollingTrainConfig, fit_rolling_gcn
from graph_ml.utils.seed import config_hash


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _aggregate(results: list[dict]) -> dict[str, float]:
    if not results:
        return {}
    keys = results[0].keys()
    return {
        k: float(np.mean([r[k] for r in results]))
        for k in keys
        if isinstance(results[0][k], (int, float))
    }


def _aggregate_std(results: list[dict]) -> dict[str, float]:
    if not results:
        return {}
    keys = results[0].keys()
    return {
        k: float(np.std([r[k] for r in results]))
        for k in keys
        if isinstance(results[0][k], (int, float))
    }


def run_link_prediction_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Run GCN / GraphSAGE link prediction sweep on SBM graphs."""
    seeds = config.get("seeds", [42])
    models = config.get("models", ["GCN", "GraphSAGE"])
    n_nodes_list = config.get("n_nodes_list", [200, 400])

    train_cfg = TrainConfig(
        epochs=config.get("epochs", 80),
        lr=config.get("lr", 0.01),
        hidden_dim=config.get("hidden_dim", 64),
        n_layers=config.get("n_layers", 2),
        dropout=config.get("dropout", 0.2),
        neg_ratio=config.get("neg_ratio", 1.0),
        hits_k=config.get("hits_k", 10),
    )

    all_results = []
    for n_nodes in n_nodes_list:
        for model_name in models:
            seed_results = []
            for seed in seeds:
                data = generate_sbm_graph(
                    SBMDGPConfig(
                        n_nodes=n_nodes,
                        n_communities=config.get("n_communities", 4),
                        feature_dim=config.get("feature_dim", 16),
                        p_in=config.get("p_in", 0.25),
                        p_out=config.get("p_out", 0.02),
                        train_ratio=config.get("train_ratio", 0.7),
                        val_ratio=config.get("val_ratio", 0.15),
                        neg_ratio=config.get("neg_ratio", 1.0),
                        seed=seed,
                    )
                )
                result = fit_link_predictor(
                    data,
                    model_name=model_name,
                    config=TrainConfig(**{**train_cfg.__dict__, "seed": seed}),
                )
                seed_results.append(
                    {
                        "val_auc": result.val_metrics.auc,
                        "val_ap": result.val_metrics.ap,
                        "val_hits_at_k": result.val_metrics.hits_at_k,
                        "test_auc": result.test_metrics.auc,
                        "test_ap": result.test_metrics.ap,
                        "test_hits_at_k": result.test_metrics.hits_at_k,
                        "train_loss": result.train_loss,
                    }
                )
            mean = _aggregate(seed_results)
            std = _aggregate_std(seed_results)
            all_results.append(
                {
                    "model": model_name,
                    "n_nodes": n_nodes,
                    **{f"{k}_mean": v for k, v in mean.items()},
                    **{f"{k}_std": v for k, v in std.items()},
                }
            )
    return {"module": "link_prediction", "results": all_results}


def run_temporal_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Run rolling GCN temporal link forecasting sweep."""
    seeds = config.get("seeds", [42])
    models = config.get("models", ["RollingGCN"])

    train_cfg = RollingTrainConfig(
        epochs=config.get("epochs", 60),
        lr=config.get("lr", 0.01),
        hidden_dim=config.get("hidden_dim", 64),
        n_layers=config.get("n_layers", 2),
        dropout=config.get("dropout", 0.2),
        history_window=config.get("history_window", 3),
        neg_ratio=config.get("neg_ratio", 1.0),
    )

    all_results = []
    for model_name in models:
        if model_name != "RollingGCN":
            continue
        seed_results = []
        for seed in seeds:
            data = generate_temporal_graph(
                TemporalDGPConfig(
                    n_nodes=config.get("n_nodes", 150),
                    n_snapshots=config.get("n_snapshots", 12),
                    feature_dim=config.get("feature_dim", 12),
                    drift_strength=config.get("drift_strength", 0.15),
                    k_neighbors=config.get("k_neighbors", 8),
                    neg_ratio=config.get("neg_ratio", 1.0),
                    seed=seed,
                )
            )
            result = fit_rolling_gcn(
                data,
                config=RollingTrainConfig(**{**train_cfg.__dict__, "seed": seed}),
            )
            seed_results.append(
                {
                    "forecast_auc": result.forecast_metrics.auc,
                    "forecast_ap": result.forecast_metrics.ap,
                    "forecast_edge_f1": result.forecast_metrics.edge_f1,
                    "train_loss": result.train_loss,
                }
            )
        mean = _aggregate(seed_results)
        std = _aggregate_std(seed_results)
        all_results.append(
            {
                "model": model_name,
                "n_nodes": config.get("n_nodes", 150),
                "n_snapshots": config.get("n_snapshots", 12),
                **{f"{k}_mean": v for k, v in mean.items()},
                **{f"{k}_std": v for k, v in std.items()},
            }
        )
    return {"module": "temporal", "results": all_results}


def run_benchmark(
    config_path: str | Path,
    module: str = "all",
    output_dir: str | Path | None = None,
) -> Path:
    """Run benchmark(s) and write results."""
    config_path = Path(config_path)
    config = load_config(config_path)
    default_path = config_path.parent / "default.yaml"
    merged = {**load_config(default_path), **config} if default_path.exists() else config

    results: dict[str, Any] = {
        "config_hash": config_hash(merged),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modules": {},
    }

    if module in ("link_prediction", "all"):
        results["modules"]["link_prediction"] = run_link_prediction_benchmark(merged)
    if module in ("temporal", "all"):
        results["modules"]["temporal"] = run_temporal_benchmark(merged)

    out = Path(output_dir or "results")
    run_dir = out / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    from graph_ml.evaluation.report import write_report

    write_report(results, run_dir / "summary.md")

    return run_dir
