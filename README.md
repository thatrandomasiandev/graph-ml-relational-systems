# Graph ML & Relational Systems

PhD-level graph ML suite covering **static link prediction**, **GNN encoders** (GCN, GraphSAGE), and **temporal graph forecasting** — all evaluated on synthetic data with known ground truth.

## Modules

| Module | Description | Key metrics |
|--------|-------------|-------------|
| **Link prediction** | GCN and GraphSAGE on SBM graphs with held-out edges | AUC, AP, Hits@K |
| **Temporal graphs** | Rolling GNN snapshots on evolving latent-space dynamics | Temporal AUC, AP, edge F1 |

## Assumptions

- **Static graphs:** Transductive link prediction; node features observed; train edges do not leak into test
- **Temporal graphs:** Causal snapshot ordering; k-NN graphs from drifting latent positions; predict held-out edges in the final snapshot
- **kNN DGP:** Each snapshot is a k-nearest-neighbor graph on drifting latent positions

## Setup

```bash
cd 06-graph-ml-relational-systems
pip install -e ".[dev]"
```

## Run benchmarks

```bash
# All modules
python scripts/run_benchmark.py --config configs/link_prediction_benchmark.yaml --module all

# Individual modules
python scripts/run_benchmark.py --config configs/link_prediction_benchmark.yaml --module link_prediction
python scripts/run_benchmark.py --config configs/temporal_graph_benchmark.yaml --module temporal
```

Results are written to `results/{timestamp}/metrics.json` and `summary.md`.

## Run tests

```bash
pytest
```

## Project layout

```
src/graph_ml/
├── data/              # SBM and temporal graph DGPs with ground-truth accessors
├── models/            # GCN, GraphSAGE, link scoring heads
├── link_prediction/   # Training and evaluation for static graphs
├── temporal/          # Rolling snapshot forecasting
└── evaluation/        # Benchmark runner and reporting
```

## Future work

- Real datasets (Cora, OGB link prediction) via the same `GraphDataset` interface
- TGN / JODIE-style continuous-time encoders
- Heterogeneous and knowledge-graph link prediction
