# Graph ML & Relational Systems

A research benchmark suite for **static link prediction**, **graph neural network representation learning**, and **temporal graph forecasting** — the three core problems in learning on relational data. All experiments use synthetic graphs with known community structure or evolving latent dynamics, enabling exact evaluation of link ranking quality and temporal generalization.

The central research question: *how do message-passing architectures exploit graph structure for predicting missing edges, and how do they adapt as relational topology evolves over time?*

---

## Research scope

| Module | Problem | Methods | Primary metrics |
|--------|---------|---------|-----------------|
| **Link prediction** | Predict missing edges from partial graph + node features | GCN, GraphSAGE + link scoring head | AUC, AP, Hits@K |
| **Temporal graphs** | Forecast edges in future graph snapshots | Rolling GNN on kNN snapshots | Temporal AUC, AP, edge F1 |

---

## Module 1: Static link prediction

### Problem formulation

Given a partially observed graph G = (V, E_obs) with node features X, predict which edges exist in the full graph. This is the **link prediction problem** (Liben-Nowell & Kleinberg, 2007): rank candidate pairs (u,v) by predicted link probability.

### Graph neural network encoders

| Architecture | Aggregation | Reference |
|-------------|-------------|-----------|
| **GCN** | Spectral graph convolution: H^(l+1) = σ(D̃^{-½}ÃD̃^{-½} H^(l) W^(l)) | Kipf & Welling (2017) |
| **GraphSAGE** | Sample-and-aggregate: inductive learning on neighborhood samples | Hamilton et al. (2017) |

GCN (Kipf & Welling, 2017) performs spectral convolutions on the graph Laplacian, propagating information along edges. GraphSAGE (Hamilton et al., 2017) samples fixed-size neighborhoods and applies learnable aggregation functions, enabling **inductive** learning on graphs not seen during training.

### Link scoring

Node embeddings h_u, h_v from the final GNN layer are combined via a **bilinear scoring head** to produce link probabilities. Training uses negative sampling: observed edges as positives, random non-edges as negatives.

### Synthetic DGP (`data/static_graph_dgp.py`)

**Stochastic block model (SBM)** (Holland et al., 1983):
- Nodes assigned to K communities
- Within-community edge probability p_in > between-community p_out
- Node features are noisy projections of community labels
- Ground-truth adjacency matrix and community assignments available

### Evaluation metrics

- **AUC / AP:** Ranking quality of positive vs. negative edge pairs
- **Hits@K:** Fraction of true test edges ranked in top-K predictions

---

## Module 2: Temporal graph forecasting

### Problem formulation

Given a sequence of graph snapshots G₁, G₂, ..., G_T, predict edges in G_{T+1}. This captures **evolving relational structure** — social networks, citation graphs, and biological interaction networks all exhibit temporal dynamics (Kazemi et al., 2020).

### Implemented method

**Rolling GNN** (`temporal/rolling_gnn.py`):
1. Train GNN encoder on snapshots G₁...G_{T-1}
2. Generate node embeddings for snapshot G_T
3. Predict edges in G_T using link scoring head
4. Retrain periodically as new snapshots arrive

### Synthetic DGP (`data/temporal_graph_dgp.py`)

- Latent positions in ℝ^d drift over time via random walk
- Each snapshot G_t is a **k-nearest-neighbor graph** on current latent positions
- Ground-truth edges known for every snapshot
- Tunable drift strength controls temporal non-stationarity

### Evaluation metrics

- **Temporal AUC / AP:** Link ranking on the held-out final snapshot
- **Edge F1:** Binary classification of edge existence at threshold

---

## Benchmark protocol

```bash
pip install -e ".[dev]"

python scripts/run_benchmark.py --config configs/link_prediction_benchmark.yaml --module all
python scripts/run_benchmark.py --config configs/link_prediction_benchmark.yaml --module link_prediction
python scripts/run_benchmark.py --config configs/temporal_graph_benchmark.yaml --module temporal

pytest
```

Configs sweep graph sizes (n ∈ {200, 400}), encoder architectures, and temporal drift strength.

---

## Project layout

```
src/graph_ml/
├── data/              # SBM and temporal kNN graph DGPs
├── models/            # GCN, GraphSAGE, link scoring heads
├── link_prediction/   # Training and evaluation for static graphs
├── temporal/          # Rolling snapshot forecasting
└── evaluation/        # Benchmark runner and reporting
```

---

## Implementation notes

- Link prediction is **transductive**: all node features are observed; only edges are masked
- Train edges do not leak into test sets (proper edge holdout)
- Temporal forecasting uses **discrete snapshots**, not continuous-time models (TGN, JODIE)
- GCN uses full-batch training suitable for graphs up to ~1000 nodes

---

## References

- Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on large graphs. *NeurIPS*. [arXiv](https://arxiv.org/abs/1706.02216)
- Holland, P. W., Laskey, K. B., & Leinhardt, S. (1983). Stochastic blockmodels: First steps. *Social Networks*, 5(2), 109–137. [DOI](https://doi.org/10.1016/0378-8733(83)90021-7)
- Kazemi, S. M., Goel, R., Eghbali, S., Ramanan, J., Sahota, J., Thakur, S., Wu, S., Smyth, C., Poupart, P., & Brubaker, M. (2020). Representation learning for dynamic graphs: A survey. *JMLR*, 21(70), 1–56. [Paper](https://jmlr.org/papers/v21/20-212.html)
- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*. [arXiv](https://arxiv.org/abs/1609.02907)
- Liben-Nowell, D., & Kleinberg, J. (2007). The link-prediction problem for social networks. *JASIST*, 58(7), 1019–1031. [DOI](https://doi.org/10.1002/asi.20591)
- Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks? *ICLR*. [arXiv](https://arxiv.org/abs/1810.00826)

---

## Future work

- Real benchmarks: Cora, OGB link prediction (Hu et al., 2020)
- Continuous-time encoders: TGN (Rossi et al., 2020), JODIE (Kumar et al., 2019)
- Heterogeneous and knowledge-graph link prediction (Schlichtkrull et al., 2018)
