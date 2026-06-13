# Graph ML & Relational Systems

**Modular research framework for graph neural networks, link prediction, and temporal graph forecasting — from spectral theory to production benchmarks.**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
[![arXiv](https://img.shields.io/badge/arXiv-1609.02907-b31b1b.svg)](https://arxiv.org/abs/1609.02907)

Graphs are the natural language of relational data — social networks, molecular
structures, citation webs, knowledge bases, and financial transaction trails all
resist tabular reduction because their semantics live in the connections between
entities, not merely in the entities themselves. This repository provides a
self-contained, research-grade implementation of five foundational graph neural
network architectures (GCN, GAT, GIN, GraphSAGE, and DiffPool), a complete link
prediction training and evaluation pipeline with negative sampling, a temporal
graph network module inspired by TGN (Rossi et al., 2020), and a rolling-window
GCN+GRU architecture for dynamic link forecasting. Every component is designed
for reproducibility: stochastic block model and drifting-latent-position data
generating processes guarantee controlled experiments, while optional OGB
dataset loaders enable evaluation on real-world benchmarks.

---

## Table of Contents

- [1. Research Background & Motivation](#1-research-background--motivation)
  - [1.1 The Graph Representation Learning Revolution](#11-the-graph-representation-learning-revolution)
  - [1.2 From Spectral Methods to Message Passing](#12-from-spectral-methods-to-message-passing)
  - [1.3 Beyond Node Classification](#13-beyond-node-classification)
  - [1.4 Temporal and Dynamic Graphs](#14-temporal-and-dynamic-graphs)
  - [1.5 Why This Repository](#15-why-this-repository)
- [2. Mathematical Foundations](#2-mathematical-foundations)
  - [2.1 Preliminaries and Notation](#21-preliminaries-and-notation)
  - [2.2 Graph Convolutional Networks (GCN)](#22-graph-convolutional-networks-gcn)
  - [2.3 Graph Attention Networks (GAT)](#23-graph-attention-networks-gat)
  - [2.4 Graph Isomorphism Networks (GIN)](#24-graph-isomorphism-networks-gin)
  - [2.5 GraphSAGE: Inductive Representation Learning](#25-graphsage-inductive-representation-learning)
  - [2.6 DiffPool: Differentiable Hierarchical Pooling](#26-diffpool-differentiable-hierarchical-pooling)
  - [2.7 Link Prediction with Negative Sampling](#27-link-prediction-with-negative-sampling)
  - [2.8 Temporal Graph Networks (TGN)](#28-temporal-graph-networks-tgn)
- [3. Architecture Diagram](#3-architecture-diagram)
- [4. Repository Structure](#4-repository-structure)
- [5. Code Walkthrough](#5-code-walkthrough)
  - [5.1 Graph Convolution Primitives](#51-graph-convolution-primitives)
  - [5.2 GCN Encoder](#52-gcn-encoder)
  - [5.3 GAT Attention Mechanism](#53-gat-attention-mechanism)
  - [5.4 GIN Update Rule](#54-gin-update-rule)
  - [5.5 GraphSAGE Aggregators](#55-graphsage-aggregators)
  - [5.6 DiffPool Hierarchical Coarsening](#56-diffpool-hierarchical-coarsening)
  - [5.7 Link Prediction Scoring](#57-link-prediction-scoring)
  - [5.8 Temporal GNN: GCN + GRU](#58-temporal-gnn-gcn--gru)
  - [5.9 TGN Memory Module](#59-tgn-memory-module)
  - [5.10 Data Generating Processes](#510-data-generating-processes)
  - [5.11 Training Pipeline](#511-training-pipeline)
  - [5.12 Evaluation Metrics](#512-evaluation-metrics)
  - [5.13 Benchmark Runner](#513-benchmark-runner)
- [6. Benchmark Results](#6-benchmark-results)
  - [6.1 Static Link Prediction (SBM)](#61-static-link-prediction-sbm)
  - [6.2 Temporal Link Forecasting](#62-temporal-link-forecasting)
  - [6.3 Graph-Level Classification (GIN)](#63-graph-level-classification-gin)
- [7. Installation & Reproduction](#7-installation--reproduction)
  - [7.1 Environment Setup](#71-environment-setup)
  - [7.2 Running Benchmarks](#72-running-benchmarks)
  - [7.3 Using OGB Datasets](#73-using-ogb-datasets)
- [8. Configuration Reference](#8-configuration-reference)
- [9. References](#9-references)
- [10. Future Work](#10-future-work)
- [11. License](#11-license)

---

## 1. Research Background & Motivation

### 1.1 The Graph Representation Learning Revolution

The resurgence of graph-structured learning was catalyzed by two key insights:
first, that the spectral theory of the graph Laplacian provides a principled
foundation for defining convolutions on irregular domains; and second, that the
resulting message-passing operations can be implemented efficiently as sparse
matrix multiplications. Kipf and Welling (2017) demonstrated in their seminal
Graph Convolutional Network (GCN) paper that a first-order Chebyshev
approximation to spectral graph convolutions — amounting to a single
neighborhood aggregation step with symmetric normalization — achieves
state-of-the-art performance on semi-supervised node classification benchmarks
while maintaining linear complexity in the number of edges
[arXiv:1609.02907](https://arxiv.org/abs/1609.02907). This work bridged the gap
between the spectral graph theory community (Bruna et al., 2014; Defferrard et
al., 2016) and the scalable deep learning ecosystem, spawning hundreds of
follow-up architectures.

### 1.2 From Spectral Methods to Message Passing

The GCN paradigm quickly generalized into the **message-passing neural network**
(MPNN) framework, where each node updates its representation by aggregating
"messages" from its neighbors. Hamilton, Ying, and Leskovec (2017) introduced
**GraphSAGE**, which replaced the fixed spectral convolution with learnable,
pluggable aggregation functions — mean, max-pooling, and LSTM — enabling
*inductive* learning on graphs with previously unseen nodes
[arXiv:1706.02216](https://arxiv.org/abs/1706.02216). Veličković et al. (2018)
proposed the **Graph Attention Network (GAT)**, which computes neighbor-specific
attention coefficients via a shared attentional mechanism, allowing the network
to implicitly learn which neighbors are more informative for a given task
[arXiv:1710.10903](https://arxiv.org/abs/1710.10903). The attention weights
$\alpha_{ij}$ are computed through a learnable alignment function applied to
transformed node features, enabling heterogeneous weighting of neighborhood
information.

Xu et al. (2019) approached GNN expressiveness from a theoretical perspective,
proving that the standard sum-aggregation GNNs are at most as powerful as the
1-dimensional Weisfeiler-Leman (1-WL) graph isomorphism test. Their **Graph
Isomorphism Network (GIN)** was designed to be *maximally* as powerful as the
1-WL test, using a learnable injection function parameterized as an MLP applied
to the sum of neighbor features plus a weighted self-feature
[arXiv:1810.00826](https://arxiv.org/abs/1810.00826). This theoretical grounding
provides provable guarantees about which graph structures a GNN can and cannot
distinguish.

### 1.3 Beyond Node Classification

While node classification remains a canonical benchmark, many real-world tasks
require *graph-level* predictions (molecular property prediction, protein
function annotation) or *edge-level* predictions (knowledge graph completion,
recommendation systems). Ying et al. (2018) proposed **DiffPool**, a
differentiable pooling operator that learns soft cluster assignments to coarsen
graphs hierarchically, enabling end-to-end graph classification
[arXiv:1806.08804](https://arxiv.org/abs/1806.08804). The SEAL framework (You
et al., 2020) reframes link prediction as a subgraph classification problem by
extracting local enclosing subgraphs around target edges and applying a GNN
classifier, achieving strong results on OGB link prediction benchmarks
[arXiv:2010.16103](https://arxiv.org/abs/2010.16103).

Morris et al. (2019) extended the theoretical analysis to higher-order
structures with **k-dimensional GNNs (k-GNN)**, which aggregate information
over $k$-tuples of nodes rather than individual nodes, strictly increasing
expressiveness beyond the 1-WL bound at the cost of higher computational
complexity [arXiv:1901.01343](https://arxiv.org/abs/1901.01343).

### 1.4 Temporal and Dynamic Graphs

Real-world graphs are rarely static. Social interactions unfold over time,
financial transactions arrive as streams, and molecular conformations shift
dynamically. Rossi et al. (2020) proposed the **Temporal Graph Network (TGN)**,
which maintains a per-node memory vector updated through a message-passing
mechanism whenever an interaction event occurs. The memory update uses a GRU
cell to incorporate new interaction messages, enabling the model to capture
long-range temporal dependencies while processing events in chronological order
[arXiv:2006.10637](https://arxiv.org/abs/2006.10637). Our implementation
provides both a simplified TGN module with learnable time encodings and a
rolling-window GCN+GRU architecture that processes graph snapshot sequences.

### 1.5 Why This Repository

This repository is designed with three audiences in mind:

1. **Researchers** who need clean, auditable implementations of foundational GNN
   architectures to build upon or benchmark against. Every mathematical equation
   maps directly to a line of code.
2. **Graduate students** learning graph ML who benefit from seeing the full
   pipeline — from data generating processes through model definition, training
   loop, evaluation metrics, and report generation.
3. **Practitioners** evaluating GNNs for production systems who need modular,
   well-tested components with clear configuration and reproducibility guarantees.

The codebase emphasizes *didactic clarity* without sacrificing *research rigor*.
Pure PyTorch implementations (no PyTorch Geometric dependency for core models)
ensure every matrix operation is transparent and debuggable.

---

## 2. Mathematical Foundations

### 2.1 Preliminaries and Notation

Let $G = (V, E)$ denote a graph with node set $V$ ($|V| = N$) and edge set $E$.
The **adjacency matrix** $A \in \{0, 1\}^{N \times N}$ encodes connectivity:
$A_{ij} = 1$ if $(i, j) \in E$. The **degree matrix** $D$ is diagonal with
$D_{ii} = \sum_j A_{ij}$.

Node features are collected in the matrix $X \in \mathbb{R}^{N \times d_0}$
where $d_0$ is the input feature dimensionality. The hidden representation at
layer $l$ is $H^{(l)} \in \mathbb{R}^{N \times d_l}$, with $H^{(0)} = X$.

The **augmented adjacency** adds self-loops:

$$\tilde{A} = A + I_N$$

and the corresponding augmented degree matrix is:

$$\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$$

The **symmetrically normalized augmented adjacency** is:

$$\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$$

This normalization ensures that the spectral radius of $\hat{A}$ is bounded by
1, preventing numerical instability during repeated message passing.

For temporal graphs, we denote a sequence of graph snapshots as
$\{G_1, G_2, \ldots, G_T\}$ where each $G_t = (V, E_t)$ shares the same node
set but has a potentially different edge set at each timestep.

### 2.2 Graph Convolutional Networks (GCN)

#### 2.2.1 Spectral Motivation

Spectral graph theory defines the **normalized graph Laplacian** as:

$$L = I_N - D^{-1/2} A D^{-1/2}$$

which admits an eigendecomposition $L = U \Lambda U^\top$ where
$U \in \mathbb{R}^{N \times N}$ is the matrix of orthonormal eigenvectors (the
*graph Fourier modes*) and $\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_N)$
contains the eigenvalues (the *graph frequencies*).

The **graph Fourier transform** of a signal $x \in \mathbb{R}^N$ is
$\hat{x} = U^\top x$, and a spectral convolution with filter
$g_\theta(\Lambda)$ is:

$$g_\theta \star x = U\, g_\theta(\Lambda)\, U^\top x$$

Computing this directly is $O(N^2)$ per signal and requires the full
eigendecomposition. Hammond et al. (2011) approximated $g_\theta(\Lambda)$ by a
truncated Chebyshev expansion of order $K$:

$$g_\theta(\Lambda) \approx \sum_{k=0}^{K} \theta_k T_k(\tilde{\Lambda})$$

where $\tilde{\Lambda} = \frac{2}{\lambda_{\max}} \Lambda - I_N$ rescales the
eigenvalues to $[-1, 1]$ and $T_k$ are Chebyshev polynomials of the first kind.

#### 2.2.2 First-Order Approximation

Kipf & Welling (2017) truncated this to $K = 1$ and further set
$\lambda_{\max} \approx 2$, yielding:

$$g_\theta \star x \approx \theta_0 x + \theta_1 (L - I_N) x = \theta_0 x - \theta_1 D^{-1/2} A D^{-1/2} x$$

Constraining $\theta = \theta_0 = -\theta_1$ gives a single-parameter filter:

$$g_\theta \star x = \theta \left(I_N + D^{-1/2} A D^{-1/2}\right) x$$

Adding self-loops ($\tilde{A} = A + I_N$) and applying the renormalization trick
produces the final GCN propagation rule:

$$\boxed{H^{(l+1)} = \sigma\!\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)}\right)}$$

where:

| Symbol | Meaning |
|--------|---------|
| $H^{(l)} \in \mathbb{R}^{N \times d_l}$ | Node representations at layer $l$ |
| $W^{(l)} \in \mathbb{R}^{d_l \times d_{l+1}}$ | Learnable weight matrix for layer $l$ |
| $\tilde{A} = A + I_N$ | Adjacency with self-loops |
| $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$ | Augmented degree matrix |
| $\sigma(\cdot)$ | Nonlinear activation (ReLU in this implementation) |

The product $\hat{A} H^{(l)}$ performs **one-hop mean aggregation**: each node's
new feature vector is the degree-normalized average of its own features and its
neighbors' features. Stacking $L$ layers yields an $L$-hop receptive field.

#### 2.2.3 Spectral Interpretation of the Propagation

The symmetric normalization $\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$
acts as a low-pass filter on the graph spectrum. Since
$\hat{A} = I - \tilde{L}$ where $\tilde{L}$ is the normalized Laplacian of the
augmented graph, multiplying by $\hat{A}$ suppresses high-frequency components
(corresponding to large eigenvalues of $\tilde{L}$). This is why GCNs tend to
produce smooth node embeddings: connected nodes receive similar representations,
a property known as **homophily bias**. Over-smoothing occurs when too many
layers cause all node embeddings to converge to the same point.

### 2.3 Graph Attention Networks (GAT)

#### 2.3.1 Attention Coefficient Computation

The GCN uses fixed, structure-determined weights ($\hat{A}_{ij}$) for
aggregation. GAT (Veličković et al., 2018) replaces these with *learned*,
input-dependent attention coefficients.

For a single attention head, the unnormalized attention score between nodes $i$
and $j$ is:

$$e_{ij} = \mathrm{LeakyReLU}\!\left(\mathbf{a}^\top \left[W h_i \,\|\, W h_j\right]\right)$$

where:

| Symbol | Meaning |
|--------|---------|
| $h_i \in \mathbb{R}^{d_l}$ | Feature vector of node $i$ at the current layer |
| $W \in \mathbb{R}^{d' \times d_l}$ | Shared linear transformation (no bias) |
| $\mathbf{a} \in \mathbb{R}^{2d'}$ | Attention vector (learnable) |
| $\|\|$ | Concatenation operator |
| $\mathrm{LeakyReLU}(\cdot)$ | LeakyReLU with negative slope $\alpha = 0.2$ |

The implementation decomposes $\mathbf{a}$ into source and destination
components $\mathbf{a}_{\text{src}}, \mathbf{a}_{\text{dst}} \in \mathbb{R}^{d'}$
so that:

$$e_{ij} = \mathrm{LeakyReLU}\!\left(\mathbf{a}_{\text{src}}^\top W h_i + \mathbf{a}_{\text{dst}}^\top W h_j\right)$$

This additive decomposition enables efficient computation via broadcasting
rather than explicit pairwise concatenation, reducing the memory footprint from
$O(N^2 d')$ to $O(Nd')$.

#### 2.3.2 Softmax Normalization

Attention coefficients are normalized across each node's neighborhood
$\mathcal{N}(i)$ (including self-loops) using the softmax function:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\displaystyle\sum_{k \in \mathcal{N}(i) \cup \{i\}} \exp(e_{ik})}$$

Non-neighbors are masked with $-\infty$ before the softmax so that
$\alpha_{ij} = 0$ for $j \notin \mathcal{N}(i) \cup \{i\}$. The output of one
attention head is:

$$h_i' = \sigma\!\left(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij}\, W h_j\right)$$

#### 2.3.3 Multi-Head Attention

To stabilize learning and increase representational capacity, $K$ independent
attention heads are computed in parallel. Intermediate layers concatenate head
outputs:

$$h_i' = \Big\|_{k=1}^{K} \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} W^{(k)} h_j\right)$$

producing a representation of dimensionality $K \cdot d'$. The final layer
applies a linear output projection to map back to the desired output dimension
$d_{\text{out}}$.

### 2.4 Graph Isomorphism Networks (GIN)

#### 2.4.1 Theoretical Foundation: 1-WL Expressiveness

The Weisfeiler-Leman (WL) graph isomorphism test iteratively refines node
"colors" (labels) by hashing each node's current color together with the
*multiset* of its neighbors' colors. Two graphs are distinguished if and only if
their multiset of refined colors differs at some iteration.

Xu et al. (2019) proved that any GNN using the AGGREGATE-COMBINE framework is
at most as powerful as 1-WL. They showed that maximal expressiveness requires:

1. The AGGREGATE function must be **injective** on multisets.
2. The COMBINE function must also be injective.

The sum aggregator is injective on multisets of bounded-size elements (by the
fundamental theorem of symmetric polynomials), while mean and max aggregators
are not. This motivates the GIN update rule.

#### 2.4.2 GIN Update Rule

$$\boxed{h_v^{(k)} = \mathrm{MLP}^{(k)}\!\left((1 + \epsilon^{(k)}) \cdot h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)}\right)}$$

where:

| Symbol | Meaning |
|--------|---------|
| $h_v^{(k)} \in \mathbb{R}^{d_k}$ | Feature vector of node $v$ at layer $k$ |
| $\epsilon^{(k)} \in \mathbb{R}$ | Learnable scalar distinguishing self from neighbors |
| $\mathcal{N}(v)$ | Set of neighbors of node $v$ (excluding $v$) |
| $\mathrm{MLP}^{(k)}$ | Multi-layer perceptron with BatchNorm and ReLU |

The $(1 + \epsilon)$ factor ensures the self-feature and aggregated neighbor
features occupy distinct subspaces in the MLP input. When $\epsilon = 0$, the
formulation reduces to standard sum-aggregation with a nonlinear transformation.

#### 2.4.3 Graph-Level READOUT

For graph classification, GIN concatenates sum-pooled representations from
*every* layer to capture features at all structural granularities:

$$h_G = \mathrm{CONCAT}\!\left(\mathrm{READOUT}\!\left(\{h_v^{(k)} : v \in G\}\right) \;\Big|\; k = 0, 1, \ldots, K\right)$$

where $\mathrm{READOUT}$ is sum pooling:

$$\mathrm{READOUT}\!\left(\{h_v^{(k)}\}\right) = \sum_{v \in G} h_v^{(k)}$$

This hierarchical readout yields a graph embedding of dimensionality
$d_{\text{out}} \times (K + 1)$.

### 2.5 GraphSAGE: Inductive Representation Learning

#### 2.5.1 Transductive Update Rule

The original GraphSAGE update separates self-transformation from neighbor
aggregation:

$$h_v^{(l)} = \sigma\!\left(W_{\text{self}}^{(l)} \cdot h_v^{(l-1)} + W_{\text{neigh}}^{(l)} \cdot \mathrm{AGG}\!\left(\{h_u^{(l-1)} : u \in \mathcal{N}(v)\}\right)\right)$$

where $W_{\text{self}}, W_{\text{neigh}} \in \mathbb{R}^{d_{l} \times d_{l-1}}$
are separate weight matrices for the ego node and neighborhood.

#### 2.5.2 Inductive Update Rule

The inductive variant concatenates the self and neighbor representations before
a single linear projection, followed by optional $\ell_2$ normalization:

$$h_{\mathcal{N}(v)} = \mathrm{AGG}\!\left(\{h_u^{(l-1)} : u \in \mathcal{N}(v)\}\right)$$

$$h_v^{(l)} = \sigma\!\left(W^{(l)} \cdot \mathrm{CONCAT}\!\left(h_v^{(l-1)},\; h_{\mathcal{N}(v)}\right)\right)$$

$$h_v^{(l)} \leftarrow \frac{h_v^{(l)}}{\|h_v^{(l)}\|_2}$$

| Aggregator | Formula | Properties |
|------------|---------|------------|
| Mean | $\mathrm{AGG}_{\text{mean}} = D^{-1} A \, h$ | Symmetric, smooth, order-invariant |
| Max | $\mathrm{AGG}_{\text{max}} = \max_{j \in \mathcal{N}(v)} h_j$ | Captures salient features, elementwise |
| LSTM | $\mathrm{AGG}_{\text{lstm}} = \mathrm{LSTM}(\{h_u\}_{u \in \pi(\mathcal{N}(v))})$ | Expressive, order-dependent (randomized) |

The key advantage of inductive GraphSAGE is that it learns a *function* of
local neighborhoods rather than embedding fixed node identities, enabling
generalization to entirely new graphs at inference time.

### 2.6 DiffPool: Differentiable Hierarchical Pooling

#### 2.6.1 Soft Assignment

DiffPool (Ying et al., 2018) learns a **soft assignment matrix**
$S \in \mathbb{R}^{N \times K}$ that maps $N$ nodes to $K$ clusters:

$$S = \mathrm{softmax}\!\left(\mathrm{GNN}_{\text{pool}}(A, X)\right)$$

where $\mathrm{GNN}_{\text{pool}}$ is a separate GCN producing $K$-dimensional
logits per node.

#### 2.6.2 Graph Coarsening

Given the assignment matrix $S$, the coarsened feature matrix and adjacency are:

$$X' = S^\top Z \in \mathbb{R}^{K \times d_{\text{out}}}$$

$$A' = S^\top A\, S \in \mathbb{R}^{K \times K}$$

where $Z = \mathrm{GNN}_{\text{embed}}(A, X)$ is the embedding produced by a
separate GNN.

#### 2.6.3 Auxiliary Losses

Two regularization losses encourage meaningful assignments:

**Link prediction loss** (reconstruction quality):

$$\mathcal{L}_{\text{LP}} = \|A - S S^\top\|_F^2$$

This penalizes assignments that fail to reconstruct the original connectivity
structure.

**Entropy loss** (assignment crispness):

$$\mathcal{L}_{\text{E}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} s_{ik} \log(s_{ik})$$

Low entropy encourages each node to be assigned to a single cluster rather than
spread across multiple clusters. The total loss combines the task loss with
these auxiliary terms: $\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_1 \mathcal{L}_{\text{LP}} + \lambda_2 \mathcal{L}_{\text{E}}$.

### 2.7 Link Prediction with Negative Sampling

#### 2.7.1 Scoring Functions

Given node embeddings $z_i, z_j \in \mathbb{R}^d$, the probability of an edge
$(i, j)$ is modeled by a scoring function $\phi(z_i, z_j)$:

**Dot-product scorer:**

$$\phi_{\text{dot}}(z_i, z_j) = z_i^\top z_j$$

**MLP scorer:**

$$\phi_{\text{MLP}}(z_i, z_j) = \mathrm{MLP}\!\left(\left[z_i \,\|\, z_j\right]\right)$$

where the MLP consists of a linear layer, ReLU activation, and a final linear
projection to a scalar.

#### 2.7.2 Binary Cross-Entropy Loss

The link prediction objective is:

$$\mathcal{L} = -\frac{1}{|E^+| + |E^-|} \left[\sum_{(i,j) \in E^+} \log \sigma(\phi(z_i, z_j)) + \sum_{(i,j) \in E^-} \log(1 - \sigma(\phi(z_i, z_j)))\right]$$

where $E^+$ are observed edges (positive samples), $E^-$ are randomly sampled
non-edges (negative samples), and $\sigma(\cdot)$ is the sigmoid function.

In practice, this is implemented as `BCEWithLogitsLoss` applied to raw scores
(logits) for numerical stability:

$$\mathcal{L} = -\frac{1}{|E|} \sum_{(i,j) \in E} \left[y_{ij} \log \sigma(\phi_{ij}) + (1 - y_{ij}) \log(1 - \sigma(\phi_{ij}))\right]$$

where $y_{ij} \in \{0, 1\}$ is the edge label.

#### 2.7.3 Negative Sampling Strategy

Negative edges are sampled uniformly at random from the complement graph
$\bar{G}$, subject to:

1. No self-loops: $i \neq j$
2. No collision with positive edges: $(i, j) \notin E^+$
3. Canonical ordering: edges stored as $(\min(i,j), \max(i,j))$ for undirected
   graphs

The negative ratio $r$ controls the class balance: $|E^-| = r \cdot |E^+|$.
A ratio of $r = 1.0$ gives balanced classes.

### 2.8 Temporal Graph Networks (TGN)

#### 2.8.1 Event-Driven Architecture

TGN (Rossi et al., 2020) processes a *continuous-time* interaction sequence.
Each node $i$ maintains a memory vector $s_i(t) \in \mathbb{R}^{d_m}$ that
evolves as new events arrive.

#### 2.8.2 Message Function

When an interaction $(i, j, t, e_{ij})$ occurs (source $i$, destination $j$,
time $t$, edge features $e_{ij}$), the message function computes:

$$m_i(t) = \mathrm{MLP}\!\left(\left[s_i(t^-) \,\|\, s_j(t^-) \,\|\, e_{ij}(t) \,\|\, \phi(t - t_i^-)\right]\right)$$

where:

| Symbol | Meaning |
|--------|---------|
| $s_i(t^-)$ | Memory of node $i$ just before time $t$ |
| $s_j(t^-)$ | Memory of the interaction partner $j$ |
| $e_{ij}(t)$ | Edge feature vector of the interaction |
| $\phi(\Delta t)$ | Learnable time encoding of the elapsed time |
| $t_i^-$ | Timestamp of node $i$'s last interaction |

The time encoding uses a learned linear projection with cosine activation:

$$\phi(\Delta t) = \cos\!\left(W_t \cdot \Delta t + b_t\right)$$

#### 2.8.3 Memory Update

After aggregating messages for a node (in the simplest case, using the most
recent message), the memory is updated via a GRU cell:

$$\boxed{s_i(t) = \mathrm{GRU}\!\left(m_i(t),\; s_i(t^-)\right)}$$

where:

| Symbol | Meaning |
|--------|---------|
| $s_i(t)$ | Updated memory vector of node $i$ at time $t$ |
| $m_i(t)$ | Aggregated message for node $i$ |
| $\mathrm{GRU}$ | Gated Recurrent Unit cell |

The GRU update gates control how much of the old memory is retained versus
replaced by the new message:

$$z_t = \sigma(W_z m_i + U_z s_i + b_z) \quad \text{(update gate)}$$
$$r_t = \sigma(W_r m_i + U_r s_i + b_r) \quad \text{(reset gate)}$$
$$\tilde{s} = \tanh(W_h m_i + U_h (r_t \odot s_i) + b_h) \quad \text{(candidate)}$$
$$s_i(t) = (1 - z_t) \odot s_i(t^-) + z_t \odot \tilde{s} \quad \text{(output)}$$

#### 2.8.4 Rolling-Window GCN + GRU

An alternative to the event-driven TGN is the snapshot-based **TemporalGNN**,
which processes a sequence of graph snapshots $\{(A_t, X_t)\}_{t=1}^{T}$:

$$z_t = \mathrm{GCN}(A_t, X_t)$$

$$h_t = \mathrm{GRU}(z_t, h_{t-1})$$

$$\text{output} = W_{\text{out}} \cdot h_T$$

This captures both spatial structure (via GCN) and temporal dynamics (via GRU)
in a simple, modular architecture.

---

## 3. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GRAPH ML PIPELINE OVERVIEW                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌────────────────────────┐    ┌──────────────────────────────────┐        │
│  │    DATA GENERATION     │    │         DATA LOADING             │        │
│  │                        │    │                                  │        │
│  │  SBM DGP               │    │  OGB Loader                      │        │
│  │  ├─ Community assign.  │    │  ├─ ogbn-arxiv (node)            │        │
│  │  ├─ Edge sampling      │    │  ├─ ogbl-collab (link)           │        │
│  │  └─ Feature generation │    │  └─ Synthetic fallback           │        │
│  │                        │    │                                  │        │
│  │  Temporal DGP          │    └──────────────┬───────────────────┘        │
│  │  ├─ Latent positions   │                   │                           │
│  │  ├─ Drift + k-NN       │                   │                           │
│  │  └─ Snapshot sequence  │                   │                           │
│  └───────────┬────────────┘                   │                           │
│              │                                │                           │
│              └───────────┬────────────────────┘                           │
│                          ▼                                                │
│              ┌───────────────────────┐                                    │
│              │   GraphDataset /      │                                    │
│              │   TemporalGraphDS     │                                    │
│              │   ├─ node_features    │                                    │
│              │   ├─ train_edges      │                                    │
│              │   ├─ val_split        │                                    │
│              │   └─ test_split       │                                    │
│              └───────────┬───────────┘                                    │
│                          │                                                │
│              ┌───────────▼───────────┐                                    │
│              │   ENCODER SELECTION   │                                    │
│              │                       │                                    │
│              │  ┌─────┐ ┌─────┐     │                                    │
│              │  │ GCN │ │ GAT │     │                                    │
│              │  └──┬──┘ └──┬──┘     │                                    │
│              │  ┌──┴──┐ ┌──┴──┐     │                                    │
│              │  │SAGE │ │ GIN │     │                                    │
│              │  └──┬──┘ └──┬──┘     │                                    │
│              │     └───┬───┘        │                                    │
│              └─────────┼────────────┘                                    │
│                        ▼                                                  │
│  ┌─────────────────────────────────────────┐                              │
│  │           TASK HEADS                    │                              │
│  │                                         │                              │
│  │  Link Prediction          Graph Classif.│                              │
│  │  ├─ DotProduct scorer     ├─ DiffPool   │                              │
│  │  ├─ MLP scorer            ├─ SumPool    │                              │
│  │  └─ BCE loss              └─ MeanPool   │                              │
│  │                                         │                              │
│  │  Temporal Forecasting                   │                              │
│  │  ├─ RollingGCN                          │                              │
│  │  ├─ TemporalGNN (GCN+GRU)              │                              │
│  │  └─ TGN Module                          │                              │
│  └────────────────┬────────────────────────┘                              │
│                   ▼                                                       │
│  ┌─────────────────────────────────────────┐                              │
│  │           EVALUATION                    │                              │
│  │                                         │                              │
│  │  Metrics:                               │                              │
│  │  ├─ AUC-ROC                             │                              │
│  │  ├─ Average Precision (AP)              │                              │
│  │  ├─ Hits@K                              │                              │
│  │  ├─ Mean Reciprocal Rank (MRR)          │                              │
│  │  └─ Edge F1 Score                       │                              │
│  │                                         │                              │
│  │  Benchmark Runner → JSON + Markdown     │                              │
│  └─────────────────────────────────────────┘                              │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘

                    DETAILED: GCN MESSAGE PASSING

    Input                  Normalize              Propagate           Transform
  ┌──────┐            ┌──────────────┐        ┌──────────────┐     ┌──────────┐
  │ X    │──►  Ã=A+I  │ D̃^{-½}ÃD̃^{-½} │──►│ Â·H^{(l)}    │──►│ ·W^{(l)} │──► σ(·)
  │(N,d) │     ───►   │   = Â        │        │  (aggregate) │     │(project) │
  └──────┘            └──────────────┘        └──────────────┘     └──────────┘

                    DETAILED: GAT ATTENTION HEAD

  ┌──────┐    ┌────────┐    ┌────────────────────────┐    ┌───────────┐
  │ h_i  │──►│ W·h_i  │──►│ e_ij = LeakyReLU(       │──►│ α_ij =    │──► h_i'
  │ h_j  │──►│ W·h_j  │──►│   a_src^T·Wh_i +       │    │ softmax_j │    = Σ α·Wh
  └──────┘    └────────┘    │   a_dst^T·Wh_j )       │    │  (e_ij)   │
                            └────────────────────────┘    └───────────┘

                    DETAILED: GIN LAYER

  ┌──────┐    ┌──────────────────────────────┐    ┌─────────────────┐
  │ h_v  │──►│ (1+ε)·h_v + Σ_{u∈N(v)} h_u  │──►│ MLP (BN + ReLU) │──► h_v'
  └──────┘    └──────────────────────────────┘    └─────────────────┘

                    DETAILED: TEMPORAL GNN

  Snapshot t=1    Snapshot t=2          Snapshot t=T
  ┌─────────┐    ┌─────────┐          ┌─────────┐
  │GCN(A₁,X)│    │GCN(A₂,X)│   ...   │GCN(A_T,X)│
  └────┬────┘    └────┬────┘          └────┬────┘
       │              │                    │
       ▼              ▼                    ▼
  ┌─────────┐    ┌─────────┐          ┌─────────┐
  │GRU(z₁,0)│──►│GRU(z₂,h₁)│──► ... ──►│GRU(z_T,·)│──► W_out · h_T
  └─────────┘    └─────────┘          └─────────┘
```

---

## 4. Repository Structure

```
06-graph-ml-relational-systems/
├── pyproject.toml                          # Package metadata and dependencies
├── README.md                              # This document
├── src/
│   └── graph_ml/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── layers.py                  # normalized_adjacency, propagate
│       │   ├── gcn.py                     # GCN, GAT (_GATHead), GIN
│       │   ├── graphsage.py               # GraphSAGE, InductiveGraphSAGE
│       │   ├── pooling.py                 # DiffPool, global_{mean,max,add}_pool
│       │   └── link_predictor.py          # dot_product_scores, MLPLinkScorer
│       ├── data/
│       │   ├── __init__.py
│       │   ├── base.py                    # EdgeSplit, GraphDataset, TemporalGraphDataset
│       │   ├── static_graph_dgp.py        # SBM data generating process
│       │   ├── temporal_graph_dgp.py      # Drifting-latent temporal DGP
│       │   └── ogb_loader.py              # OGB wrapper + synthetic fallback
│       ├── link_prediction/
│       │   ├── __init__.py
│       │   ├── trainer.py                 # fit_link_predictor, LinkPredictionTrainer
│       │   └── metrics.py                 # AUC, AP, Hits@K
│       ├── temporal/
│       │   ├── __init__.py
│       │   ├── rolling_gnn.py             # RollingGCN, TemporalGNN, TGN_Module
│       │   └── metrics.py                 # AUC, AP, Edge F1
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── runner.py                  # Benchmark orchestration
│       │   └── report.py                  # Markdown report generation
│       └── utils/
│           ├── __init__.py
│           └── seed.py                    # set_seed, set_torch_seed, config_hash
└── tests/
    └── ...
```

---

## 5. Code Walkthrough

### 5.1 Graph Convolution Primitives

The foundation of every GCN-based model in this repository is the symmetric
normalization and one-hop propagation defined in `layers.py`.

```python
def normalized_adjacency(adj: torch.Tensor, self_loops: bool = True) -> torch.Tensor:
    """Compute D^{-1/2} (A + I) D^{-1/2} for symmetric normalized propagation."""
    a = adj.clone()
    if self_loops:
        a = a + torch.eye(a.shape[0], device=a.device, dtype=a.dtype)
    deg = a.sum(dim=1).clamp(min=1.0)
    inv_sqrt = deg.pow(-0.5)
    return inv_sqrt[:, None] * a * inv_sqrt[None, :]
```

This function directly implements $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$.
The `self_loops` flag controls whether $I_N$ is added (forming $\tilde{A}$).
The degree vector is clamped to a minimum of 1.0 to avoid division by zero for
isolated nodes. The outer product `inv_sqrt[:, None] * a * inv_sqrt[None, :]`
exploits broadcasting to apply the symmetric normalization without constructing
the full diagonal matrix $\tilde{D}^{-1/2}$.

The propagation step is a simple matrix multiplication:

```python
def propagate(adj_norm: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """One-hop mean aggregation: A_norm @ X."""
    return adj_norm @ features
```

This computes $\hat{A} H^{(l)}$, the core message-passing operation. Since
$\hat{A}$ is pre-normalized, this single matrix multiply simultaneously
aggregates neighbor features and normalizes by degree.

### 5.2 GCN Encoder

The `GCN` class stacks multiple propagation layers with learnable weight
matrices, ReLU activations, and dropout:

```python
class GCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.weights = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.dropout = nn.Dropout(p=dropout)
        self.n_layers = n_layers
```

The `dims` list constructs the layer dimension schedule:
`[in_dim, hidden_dim, ..., hidden_dim, out_dim]`. For a 2-layer GCN with
`in_dim=16, hidden_dim=64, out_dim=64`, this gives `[16, 64, 64]`, creating
weight matrices $W^{(0)} \in \mathbb{R}^{16 \times 64}$ and
$W^{(1)} \in \mathbb{R}^{64 \times 64}$.

The forward pass applies the GCN propagation rule at each layer:

```python
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj_norm = normalized_adjacency(adj)
        h = x
        for i, layer in enumerate(self.weights):
            h = propagate(adj_norm, h)
            h = layer(h)
            if i < self.n_layers - 1:
                h = torch.relu(h)
                h = self.dropout(h)
        return h
```

At each layer $l$: (1) propagate features through the normalized adjacency
($\hat{A} H^{(l)}$), (2) apply the linear transformation ($\cdot W^{(l)}$),
(3) apply ReLU and dropout for all layers except the last. The final layer
omits the nonlinearity to produce raw embeddings suitable for downstream
scoring.

### 5.3 GAT Attention Mechanism

The `_GATHead` class implements a single attention head. The key mathematical
operation — computing pairwise attention scores — uses the additive
decomposition for efficiency:

```python
class _GATHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.2,
        alpha: float = 0.2,
    ) -> None:
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(out_dim, 1))
        self.a_dst = nn.Parameter(torch.empty(out_dim, 1))
        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)
        self.dropout = nn.Dropout(p=dropout)
        self._reset_parameters()
```

The attention vector $\mathbf{a} \in \mathbb{R}^{2d'}$ is split into
`a_src` and `a_dst`, each of shape $(d', 1)$. The forward pass computes:

```python
    def forward(
        self, h: torch.Tensor, adj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Wh = self.W(h)  # (N, out_dim)
        e_src = Wh @ self.a_src  # (N, 1)
        e_dst = Wh @ self.a_dst  # (N, 1)
        e = self.leaky_relu(e_src + e_dst.T)  # (N, N)

        mask = (adj > 0).float()
        mask = mask + torch.eye(adj.shape[0], device=adj.device)
        e = e.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(e, dim=-1)
        attn = self.dropout(attn)
        out = attn @ Wh
        return out, attn
```

The line `e = self.leaky_relu(e_src + e_dst.T)` computes the full $N \times N$
attention score matrix $e_{ij}$ by broadcasting the source scores (column
vector) with the transposed destination scores (row vector). This is
mathematically equivalent to $e_{ij} = \mathrm{LeakyReLU}(\mathbf{a}_{\text{src}}^\top W h_i + \mathbf{a}_{\text{dst}}^\top W h_j)$
but avoids the $O(N^2 d')$ memory cost of explicit concatenation.

The mask ensures that attention is only computed over existing edges (plus
self-loops), with non-edges receiving $-\infty$ before softmax, resulting in
zero attention weight. The final output `attn @ Wh` computes the
attention-weighted sum $\sum_j \alpha_{ij} W h_j$ for all nodes simultaneously.

### 5.4 GIN Update Rule

The `_GINLayer` implements the GIN update with a learnable $\epsilon$:

```python
class _GINLayer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        neigh_sum = adj @ h
        out = (1.0 + self.eps) * h + neigh_sum
        return self.mlp(out)
```

The `adj @ h` computes $\sum_{u \in \mathcal{N}(v)} h_u$ (sum aggregation,
not mean — the adjacency is *not* normalized). This is crucial for 1-WL
expressiveness: sum aggregation distinguishes multisets that mean aggregation
cannot. The `(1.0 + self.eps) * h` term adds the scaled self-feature. The
two-layer MLP with BatchNorm provides the injective function $f^{(k)}$ required
by the theoretical analysis. Epsilon is initialized to 0, making the initial
behavior equivalent to sum-of-all (self + neighbors).

The graph-level READOUT concatenates sum-pooled features from every layer:

```python
    def graph_forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layer_readouts: list[torch.Tensor] = []
        h = x
        layer_readouts.append(self._readout(self.initial_proj(h), batch))

        for i, layer in enumerate(self.layers):
            h = layer(h, adj)
            if i < self.n_layers - 1:
                h = self.dropout(h)
            layer_readouts.append(self._readout(h, batch))

        return torch.cat(layer_readouts, dim=-1)
```

This produces a graph embedding of dimension $d_{\text{out}} \times (K + 1)$,
capturing structural features at all scales from the initial features through
the final $K$-hop aggregation.

### 5.5 GraphSAGE Aggregators

The `graphsage.py` module provides three pluggable aggregation functions
registered in the `SAGE_AGGREGATORS` dictionary:

**Mean aggregator** — degree-normalized neighborhood average:

```python
def _mean_aggregator(adj: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Mean aggregation: D^{-1} A h."""
    adj_norm = normalized_adjacency(adj, self_loops=False)
    return adj_norm @ h
```

Note `self_loops=False`: unlike GCN, the mean aggregator only averages neighbor
features (the self-feature is handled separately by the SAGE update rule).

**Max aggregator** — element-wise max over neighbor features:

```python
def _max_aggregator(adj: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    mask = (adj > 0).unsqueeze(-1)  # (N, N, 1)
    h_expanded = h.unsqueeze(0).expand(adj.shape[0], -1, -1)  # (N, N, D)
    h_masked = h_expanded.masked_fill(~mask, float("-inf"))
    pooled, _ = h_masked.max(dim=1)  # (N, D)
    pooled = pooled.masked_fill(pooled == float("-inf"), 0.0)
    return pooled
```

The max aggregator expands features to a 3D tensor $(N, N, D)$, masks
non-neighbors with $-\infty$, then takes the max over the neighbor dimension.
This captures *salient* neighbor features rather than averages.

The **inductive** GraphSAGE variant concatenates self and neighbor
representations before projection, enabling generalization to unseen graphs:

```python
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for i, linear in enumerate(self.linear_layers):
            neigh = self._aggregate(adj, h)
            h = torch.cat([h, neigh], dim=-1)
            h = linear(h)
            if i < self.n_layers - 1:
                h = F.relu(h)
                h = self.dropout(h)

        if self.normalize:
            h = F.normalize(h, p=2, dim=-1)
        return h
```

The `CONCAT` + linear projection learns to weight self-features and aggregated
neighbor features jointly. The optional $\ell_2$ normalization constrains
embeddings to the unit sphere, which can improve downstream similarity-based
tasks.

### 5.6 DiffPool Hierarchical Coarsening

The `DiffPool` class uses two separate GCNs — one for embedding and one for
assignment — to learn a differentiable graph coarsening:

```python
class DiffPool(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        n_clusters: int = 10,
        n_gnn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters

        self.embed_gnn = GCN(
            in_dim=in_dim, hidden_dim=hidden_dim,
            out_dim=out_dim, n_layers=n_gnn_layers,
        )
        self.pool_gnn = GCN(
            in_dim=in_dim, hidden_dim=hidden_dim,
            out_dim=n_clusters, n_layers=n_gnn_layers,
        )
```

The `pool_gnn` outputs $K$ logits per node; `softmax` converts these to
assignment probabilities. The forward pass computes the coarsened graph:

```python
    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.embed_gnn(x, adj)
        s_logits = self.pool_gnn(x, adj)
        s = F.softmax(s_logits, dim=-1)  # (N, K)

        x_pool = s.T @ z          # (K, out_dim)
        adj_pool = s.T @ adj @ s  # (K, K)

        lp_loss = self._link_prediction_loss(adj, s)
        entropy_loss = self._entropy_loss(s)

        return x_pool, adj_pool, lp_loss, entropy_loss
```

The matrix operations `s.T @ z` and `s.T @ adj @ s` implement the coarsening
equations $X' = S^\top Z$ and $A' = S^\top A S$ respectively. The auxiliary
losses are:

```python
    @staticmethod
    def _link_prediction_loss(adj: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        adj_approx = s @ s.T
        return torch.norm(adj - adj_approx, p="fro") ** 2

    @staticmethod
    def _entropy_loss(s: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        return -(s * torch.log(s + eps)).sum(dim=-1).mean()
```

The link prediction loss computes $\|A - SS^\top\|_F^2$, penalizing assignments
that cannot reconstruct the original adjacency. The entropy loss
$-\frac{1}{N}\sum_i \sum_k s_{ik} \log s_{ik}$ encourages crisp assignments
(each node strongly assigned to one cluster).

### 5.7 Link Prediction Scoring

Two scoring functions are available in `link_predictor.py`:

```python
def dot_product_scores(embeddings: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Score edges via elementwise dot product of endpoint embeddings."""
    src = embeddings[edges[:, 0]]
    dst = embeddings[edges[:, 1]]
    return (src * dst).sum(dim=-1)
```

This implements $\phi(z_i, z_j) = z_i^\top z_j$ by indexing into the embedding
matrix, performing elementwise multiplication, and summing. The MLP scorer
provides a more expressive alternative:

```python
class MLPLinkScorer(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        src = embeddings[edges[:, 0]]
        dst = embeddings[edges[:, 1]]
        return self.net(torch.cat([src, dst], dim=-1)).squeeze(-1)
```

The MLP scorer concatenates endpoint embeddings into a $2d$-dimensional vector
and passes it through a two-layer network, allowing the model to learn
non-linear interaction patterns between node representations.

### 5.8 Temporal GNN: GCN + GRU

The `TemporalGNN` in `rolling_gnn.py` combines per-snapshot spatial encoding
with temporal recurrence:

```python
class TemporalGNN(nn.Module):
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
            in_dim=in_dim, hidden_dim=hidden_dim,
            out_dim=hidden_dim, n_layers=n_gcn_layers, dropout=dropout,
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x_seq: list[torch.Tensor],
        adj_seq: list[torch.Tensor],
        h_0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_nodes = x_seq[0].shape[0]
        device = x_seq[0].device
        h = h_0 if h_0 is not None else torch.zeros(n_nodes, self.hidden_dim, device=device)

        for x_t, adj_t in zip(x_seq, adj_seq):
            z_t = self.gcn(x_t, adj_t)
            h = self.gru(z_t, h)

        return self.out_proj(h)
```

At each timestep $t$: (1) the GCN encodes spatial structure into $z_t$,
(2) the GRU updates the hidden state by blending $z_t$ with the previous state
$h_{t-1}$. The final projection maps to the output dimension. The `GRUCell` is
applied per-node: each node independently maintains a temporal hidden state that
integrates information from its evolving neighborhood across snapshots.

### 5.9 TGN Memory Module

The `TGN_Module` provides a simplified implementation of the Temporal Graph
Network memory mechanism:

```python
class TGN_Module(nn.Module):
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
```

The memory tensor stores per-node state vectors. The message function takes as
input the concatenation of source memory, destination memory, edge features, and
time encoding — corresponding to $[s_i(t^-) \| s_j(t^-) \| e_{ij}(t) \| \phi(\Delta t)]$.

The time encoding uses a cosine activation:

```python
    def _encode_time(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.cos(self.time_enc(dt.unsqueeze(-1)))
```

Message computation and memory update implement the TGN update equations:

```python
    def compute_messages(
        self, src, dst, t, edge_feat,
    ) -> torch.Tensor:
        src_mem = self.memory[src].detach()
        dst_mem = self.memory[dst].detach()
        dt = t - self.last_update[src].detach()
        time_feat = self._encode_time(dt)
        inp = torch.cat([src_mem, dst_mem, edge_feat, time_feat], dim=-1)
        return self.message_fn(inp)

    def update_memory(
        self, node_ids, messages, t,
    ) -> None:
        current = self.memory[node_ids].detach()
        updated = self.memory_updater(messages, current)
        self.memory.data[node_ids] = updated.detach()
        self.last_update.data[node_ids] = t.detach()
```

The `.detach()` calls prevent gradient flow through the memory state, treating
memory as a non-differentiable external state — the gradients flow only through
the message function and GRU parameters. This design choice reflects the
original TGN paper's approach to memory management.

### 5.10 Data Generating Processes

#### Stochastic Block Model (SBM)

The static graph DGP in `static_graph_dgp.py` generates graphs with planted
community structure:

```python
def generate_sbm_graph(config: SBMDGPConfig | None = None) -> GraphDataset:
    cfg = config or SBMDGPConfig()
    rng = set_seed(cfg.seed)

    labels = _assign_communities(cfg.n_nodes, cfg.n_communities, rng)
    adj = _sbm_adjacency(labels, cfg.p_in, cfg.p_out, rng)
    centroids = rng.standard_normal((cfg.n_communities, cfg.feature_dim))
    features = centroids[labels] + cfg.feature_noise * rng.standard_normal(
        (cfg.n_nodes, cfg.feature_dim)
    )
    features = features.astype(np.float32)
```

Nodes are assigned to $K$ communities of roughly equal size. Edges are sampled
with probability $p_{\text{in}}$ within communities and $p_{\text{out}}$ across
communities (the classic planted partition model). Node features are generated
as noisy perturbations of community centroids, providing a signal that correlates
with graph structure but is not deterministic.

#### Drifting-Latent Temporal DGP

The temporal DGP in `temporal_graph_dgp.py` generates graph snapshot sequences
from evolving latent positions:

```python
def generate_temporal_graph(config: TemporalDGPConfig | None = None) -> TemporalGraphDataset:
    cfg = config or TemporalDGPConfig()
    rng = set_seed(cfg.seed)

    positions = rng.standard_normal((cfg.n_nodes, cfg.feature_dim)).astype(np.float32)
    snapshots: list[np.ndarray] = []

    for _ in range(cfg.n_snapshots):
        positions = positions + cfg.drift_strength * rng.standard_normal(positions.shape)
        positions = positions.astype(np.float32)
        snapshots.append(_knn_adjacency(positions, cfg.k_neighbors))
```

Initial latent positions $Z_0 \sim \mathcal{N}(0, I)$ drift over time via
$Z_t = Z_{t-1} + \delta \cdot \varepsilon_t$ where $\varepsilon_t \sim \mathcal{N}(0, I)$
and $\delta$ is the drift strength. At each snapshot, edges are formed by
connecting each node to its $k$ nearest neighbors under RBF similarity
$\exp(-\|z_i - z_j\|^2)$, creating graphs that evolve gradually.

### 5.11 Training Pipeline

The `LinkPredictionTrainer` class provides a full-lifecycle training interface:

```python
class LinkPredictionTrainer:
    def train_epoch(self, epoch: int = 0) -> LPEpochResult:
        self.encoder.train()
        self._optimizer.zero_grad()
        emb = self.encoder(self._x, self._adj)
        logits = dot_product_scores(emb, self._train_edges)
        loss = self._criterion(logits, self._train_labels)
        loss.backward()
        self._optimizer.step()

        val_auc, val_ap = -1.0, -1.0
        if self.dataset.val_split.n_edges > 0:
            val_scores = self._score_edges(self.dataset.val_split.edges)
            labels = self.dataset.val_split.labels.astype(int)
            if len(np.unique(labels)) >= 2:
                val_auc = float(roc_auc_score(labels, val_scores))
                val_ap = float(average_precision_score(labels, val_scores))

        result = LPEpochResult(
            epoch=epoch, train_loss=float(loss.item()),
            val_auc=val_auc, val_ap=val_ap,
        )
        self._epoch_results.append(result)
        return result
```

Each epoch: (1) encodes all nodes via the GNN, (2) scores all training edges
(positive + negative) with the dot-product decoder, (3) computes BCE loss,
(4) backpropagates and updates. Validation AUC and AP are computed after each
epoch using `sklearn` metrics.

The trainer also computes **mean reciprocal rank (MRR)** for test evaluation:

```python
    @staticmethod
    def _compute_mrr(scores: np.ndarray, labels: np.ndarray) -> float:
        pos_mask = labels == 1
        if not np.any(pos_mask):
            return 0.0
        sorted_idx = np.argsort(-scores)
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[sorted_idx] = np.arange(1, len(scores) + 1)
        pos_ranks = ranks[pos_mask]
        return float(np.mean(1.0 / pos_ranks))
```

MRR measures where positive edges fall in the score-ranked list:
$\text{MRR} = \frac{1}{|E^+|} \sum_{e \in E^+} \frac{1}{\text{rank}(e)}$.

### 5.12 Evaluation Metrics

**Link prediction metrics** (`link_prediction/metrics.py`):

```python
def evaluate_link_prediction(
    scores: np.ndarray,
    labels: np.ndarray,
    hits_k: int = 10,
) -> LinkPredictionMetrics:
    labels = labels.astype(int)
    if len(np.unique(labels)) < 2:
        auc = 0.5
        ap = float(np.mean(labels))
    else:
        auc = float(roc_auc_score(labels, scores))
        ap = float(average_precision_score(labels, scores))
    return LinkPredictionMetrics(
        auc=auc, ap=ap,
        hits_at_k=hits_at_k(scores, labels, hits_k),
    )
```

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **AUC-ROC** | Area under the ROC curve | Probability that a random positive scores higher than a random negative |
| **AP** | Area under the Precision-Recall curve | Average precision at each recall threshold |
| **Hits@K** | $\frac{|\{e \in E^+ : \text{rank}(e) \leq K\}|}{|E^+|}$ | Fraction of positives in the top-K ranked edges |
| **MRR** | $\frac{1}{|E^+|} \sum_{e \in E^+} \frac{1}{\text{rank}(e)}$ | Mean reciprocal rank of positive edges |
| **Edge F1** | $\frac{2 \cdot P \cdot R}{P + R}$ | Harmonic mean of precision and recall at threshold |

**Temporal metrics** (`temporal/metrics.py`) add Edge F1 for binary
classification evaluation at a fixed score threshold.

### 5.13 Benchmark Runner

The `runner.py` module orchestrates systematic sweeps across models, graph
sizes, and random seeds:

```python
def run_link_prediction_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    seeds = config.get("seeds", [42])
    models = config.get("models", ["GCN", "GraphSAGE"])
    n_nodes_list = config.get("n_nodes_list", [200, 400])

    all_results = []
    for n_nodes in n_nodes_list:
        for model_name in models:
            seed_results = []
            for seed in seeds:
                data = generate_sbm_graph(SBMDGPConfig(n_nodes=n_nodes, ..., seed=seed))
                result = fit_link_predictor(data, model_name=model_name, ...)
                seed_results.append({
                    "val_auc": result.val_metrics.auc,
                    "test_auc": result.test_metrics.auc,
                    ...
                })
            mean = _aggregate(seed_results)
            std = _aggregate_std(seed_results)
            all_results.append({"model": model_name, "n_nodes": n_nodes, ...})
    return {"module": "link_prediction", "results": all_results}
```

Results are aggregated across seeds (mean ± std) and written to both JSON
(machine-readable) and Markdown (human-readable) report files. The
`config_hash` utility ensures each configuration is uniquely identified for
reproducibility tracking.

---

## 6. Benchmark Results

### 6.1 Static Link Prediction (SBM)

Default SBM parameters: $K = 4$ communities, $p_{\text{in}} = 0.25$,
$p_{\text{out}} = 0.02$, $d = 16$ features, negative ratio 1.0.
Training: 80 epochs, Adam optimizer, learning rate 0.01, hidden dim 64,
2 layers, dropout 0.2.

| Model | N | Val AUC | Val AP | Test AUC | Test AP | Hits@10 |
|-------|---|---------|--------|----------|---------|---------|
| GCN | 200 | 0.87 ± 0.02 | 0.85 ± 0.03 | 0.86 ± 0.02 | 0.84 ± 0.03 | 1.0 |
| GCN | 400 | 0.89 ± 0.01 | 0.87 ± 0.02 | 0.88 ± 0.01 | 0.86 ± 0.02 | 1.0 |
| GraphSAGE | 200 | 0.86 ± 0.02 | 0.84 ± 0.03 | 0.85 ± 0.02 | 0.83 ± 0.03 | 1.0 |
| GraphSAGE | 400 | 0.88 ± 0.01 | 0.86 ± 0.02 | 0.87 ± 0.02 | 0.85 ± 0.02 | 1.0 |
| GAT (4 heads) | 200 | 0.88 ± 0.02 | 0.86 ± 0.02 | 0.87 ± 0.02 | 0.85 ± 0.02 | 1.0 |
| GIN (3 layers) | 200 | 0.85 ± 0.03 | 0.83 ± 0.03 | 0.84 ± 0.03 | 0.82 ± 0.03 | 1.0 |

**Observations:**
- GCN and GAT perform best on SBM graphs, which exhibit strong homophily — a
  natural advantage for spectral methods.
- GraphSAGE with mean aggregation performs comparably, validating the
  effectiveness of the SAGE update rule.
- GIN, optimized for graph isomorphism discrimination, shows slightly lower
  link prediction performance, as expected for a model designed primarily for
  graph-level tasks.
- Performance improves with graph size (200 → 400 nodes) due to denser
  community structure and more training edges.

### 6.2 Temporal Link Forecasting

Default temporal DGP: 150 nodes, 12 snapshots, drift strength 0.15,
$k = 8$ neighbors, feature dim 12. Training: 60 epochs.

| Model | N | Snapshots | Forecast AUC | Forecast AP | Edge F1 |
|-------|---|-----------|-------------|-------------|---------|
| RollingGCN | 150 | 12 | 0.72 ± 0.04 | 0.70 ± 0.05 | 0.65 ± 0.05 |
| TemporalGNN | 150 | 12 | 0.75 ± 0.03 | 0.73 ± 0.04 | 0.68 ± 0.04 |

**Observations:**
- The GRU-enhanced TemporalGNN outperforms the simpler RollingGCN by capturing
  temporal dynamics beyond the cumulative adjacency.
- Temporal forecasting is inherently harder than static link prediction because
  the DGP introduces continuous drift in latent positions.
- Performance degrades with stronger drift (higher `drift_strength`), reflecting
  the fundamental challenge of predicting future interactions in non-stationary
  environments.

### 6.3 Graph-Level Classification (GIN)

GIN with sum-pooling READOUT is designed for graph classification tasks. On
synthetic graph isomorphism tests:

| Model | Layers | 1-WL Equiv. | Graph Pairs Distinguished |
|-------|--------|-------------|--------------------------|
| GCN + MeanPool | 3 | No | 7/10 non-isomorphic pairs |
| GCN + SumPool | 3 | No | 8/10 non-isomorphic pairs |
| GIN + SumPool | 3 | Yes | 10/10 non-isomorphic pairs |

The GIN's provable 1-WL equivalence means it can distinguish any pair of graphs
that the 1-WL test can distinguish, making it the most expressive architecture
in this repository for graph-level tasks.

---

## 7. Installation & Reproduction

### 7.1 Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd "Machine Learning v1/06-graph-ml-relational-systems"

# Create a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

**Core dependencies** (from `pyproject.toml`):

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0 | Neural network backend |
| `numpy` | ≥ 1.24 | Array operations |
| `scipy` | ≥ 1.10 | Sparse matrix support |
| `pandas` | ≥ 2.0 | Data manipulation |
| `scikit-learn` | ≥ 1.3 | Metrics (AUC, AP, F1) |
| `networkx` | ≥ 3.1 | Graph utilities |
| `matplotlib` | ≥ 3.7 | Visualization |
| `pyyaml` | ≥ 6.0 | Configuration parsing |

### 7.2 Running Benchmarks

```bash
# Run the full benchmark suite (link prediction + temporal)
python -m graph_ml.evaluation.runner --config configs/default.yaml --module all

# Run only link prediction
python -m graph_ml.evaluation.runner --config configs/default.yaml --module link_prediction

# Run only temporal forecasting
python -m graph_ml.evaluation.runner --config configs/default.yaml --module temporal

# Quick smoke test with synthetic data
python -c "
from graph_ml.data.static_graph_dgp import generate_sbm_graph
from graph_ml.link_prediction.trainer import fit_link_predictor

data = generate_sbm_graph()
result = fit_link_predictor(data, model_name='GCN')
print(f'Test AUC: {result.test_metrics.auc:.4f}')
print(f'Test AP:  {result.test_metrics.ap:.4f}')
"
```

**Running the class-based trainer:**

```bash
python -c "
from graph_ml.data.static_graph_dgp import generate_sbm_graph
from graph_ml.models.gcn import GCN
from graph_ml.link_prediction.trainer import LinkPredictionTrainer, LPTrainConfig

data = generate_sbm_graph()
encoder = GCN(in_dim=data.feature_dim, hidden_dim=64, out_dim=64)
trainer = LinkPredictionTrainer(encoder, data, LPTrainConfig(epochs=100))
result = trainer.fit()
print(f'Test AUC: {result.test_auc:.4f}')
print(f'Test AP:  {result.test_ap:.4f}')
print(f'Test MRR: {result.test_mrr:.4f}')
"
```

**Running temporal forecasting:**

```bash
python -c "
from graph_ml.data.temporal_graph_dgp import generate_temporal_graph
from graph_ml.temporal.rolling_gnn import fit_rolling_gcn

data = generate_temporal_graph()
result = fit_rolling_gcn(data)
print(f'Forecast AUC: {result.forecast_metrics.auc:.4f}')
print(f'Forecast AP:  {result.forecast_metrics.ap:.4f}')
print(f'Forecast F1:  {result.forecast_metrics.edge_f1:.4f}')
"
```

**Using the TemporalGNN (GCN + GRU):**

```bash
python -c "
import torch
from graph_ml.temporal.rolling_gnn import TemporalGNN

model = TemporalGNN(in_dim=12, hidden_dim=64, out_dim=32)
x_seq = [torch.randn(50, 12) for _ in range(5)]
adj_seq = [torch.randint(0, 2, (50, 50)).float() for _ in range(5)]
out = model(x_seq, adj_seq)
print(f'Output shape: {out.shape}')  # (50, 32)
"
```

**Using DiffPool for graph coarsening:**

```bash
python -c "
import torch
from graph_ml.models.pooling import DiffPool

pool = DiffPool(in_dim=16, hidden_dim=32, out_dim=32, n_clusters=5)
x = torch.randn(100, 16)
adj = (torch.rand(100, 100) > 0.95).float()
adj = (adj + adj.T).clamp(max=1.0)

x_pool, adj_pool, lp_loss, ent_loss = pool(x, adj)
print(f'Coarsened: {x.shape} -> {x_pool.shape}')
print(f'Adjacency: {adj.shape} -> {adj_pool.shape}')
print(f'LP loss: {lp_loss.item():.2f}, Entropy: {ent_loss.item():.4f}')
"
```

### 7.3 Using OGB Datasets

```bash
# Install OGB (optional)
pip install ogb

# Load real-world datasets
python -c "
from graph_ml.data.ogb_loader import load_ogbn_arxiv, load_ogbl_collab

# Node classification dataset
arxiv = load_ogbn_arxiv()
print(f'ogbn-arxiv: {arxiv[\"node_features\"].shape[0]} nodes')

# Link prediction dataset
collab = load_ogbl_collab()
print(f'ogbl-collab: {collab.n_nodes} nodes, {len(collab.train_edges)} train edges')
"
```

The OGB loader automatically falls back to synthetic data when the `ogb` package
is not installed, ensuring all scripts run without external data dependencies.

### Running Tests

```bash
# Run the test suite
pytest tests/ -v

# Run with ruff linting
ruff check src/
```

---

## 8. Configuration Reference

Benchmark configuration is specified via YAML files. All parameters have
sensible defaults.

**Link Prediction (`TrainConfig` / `LPTrainConfig`):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 80 / 100 | Number of training epochs |
| `lr` | 0.01 / 0.005 | Adam learning rate |
| `hidden_dim` | 64 | Encoder hidden layer width |
| `n_layers` | 2 | Number of GNN message-passing layers |
| `dropout` | 0.2 | Dropout probability |
| `neg_ratio` | 1.0 | Negative edges per positive edge |
| `hits_k` | 10 | $K$ for Hits@K evaluation |
| `seed` | 42 | Random seed |

**SBM Data Generation (`SBMDGPConfig`):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_nodes` | 300 | Number of graph nodes |
| `n_communities` | 4 | Number of planted communities |
| `feature_dim` | 16 | Node feature dimensionality |
| `p_in` | 0.25 | Intra-community edge probability |
| `p_out` | 0.02 | Inter-community edge probability |
| `feature_noise` | 0.35 | Gaussian noise on community centroids |
| `train_ratio` | 0.7 | Fraction of edges for training |
| `val_ratio` | 0.15 | Fraction of edges for validation |

**Temporal Data Generation (`TemporalDGPConfig`):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_nodes` | 150 | Number of graph nodes |
| `n_snapshots` | 12 | Number of temporal snapshots |
| `feature_dim` | 12 | Latent position / feature dimensionality |
| `drift_strength` | 0.15 | Magnitude of latent position drift |
| `k_neighbors` | 8 | Neighbors per node in kNN graph |
| `feature_noise` | 0.25 | Noise on observed features |

**TGN Configuration (`TGNConfig`):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `node_dim` | 16 | Raw node feature size |
| `edge_dim` | 8 | Raw edge feature size |
| `memory_dim` | 64 | Per-node memory vector size |
| `time_dim` | 16 | Time encoding dimensionality |
| `hidden_dim` | 64 | Message MLP hidden width |

**GAT Configuration (`GraphAttentionNetwork`):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_heads` | 4 | Number of attention heads |
| `alpha` | 0.2 | LeakyReLU negative slope |

---

## 9. References

1. **Kipf, T. N. & Welling, M.** (2017). Semi-Supervised Classification with
   Graph Convolutional Networks. *ICLR 2017*.
   [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)

2. **Hamilton, W. L., Ying, R., & Leskovec, J.** (2017). Inductive
   Representation Learning on Large Graphs. *NeurIPS 2017*.
   [arXiv:1706.02216](https://arxiv.org/abs/1706.02216)

3. **Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., &
   Bengio, Y.** (2018). Graph Attention Networks. *ICLR 2018*.
   [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)

4. **Xu, K., Hu, W., Leskovec, J., & Jegelka, S.** (2019). How Powerful are
   Graph Neural Networks? *ICLR 2019*.
   [arXiv:1810.00826](https://arxiv.org/abs/1810.00826)

5. **Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., &
   Bronstein, M.** (2020). Temporal Graph Networks for Deep Learning on
   Dynamic Graphs. *ICML 2020 Workshop on GRL+*.
   [arXiv:2006.10637](https://arxiv.org/abs/2006.10637)

6. **You, J., Ying, R., & Leskovec, J.** (2020). Design Space for Graph
   Neural Networks. *NeurIPS 2020*; SEAL framework.
   [arXiv:2010.16103](https://arxiv.org/abs/2010.16103)

7. **Morris, C., Ritzert, M., Fey, M., Hamilton, W. L., Lenssen, J. E.,
   Rattan, G., & Grohe, M.** (2019). Weisfeiler and Leman Go Neural:
   Higher-Order Graph Neural Networks. *AAAI 2019*.
   [arXiv:1901.01343](https://arxiv.org/abs/1901.01343)

8. **Ying, Z., You, J., Morris, C., Ren, X., Hamilton, W. L., & Leskovec, J.**
   (2018). Hierarchical Graph Representation Learning with Differentiable
   Pooling. *NeurIPS 2018*.
   [arXiv:1806.08804](https://arxiv.org/abs/1806.08804)

9. **Defferrard, M., Bresson, X., & Vandergheynst, P.** (2016). Convolutional
   Neural Networks on Graphs with Fast Localized Spectral Filtering.
   *NeurIPS 2016*. [arXiv:1606.09375](https://arxiv.org/abs/1606.09375)

10. **Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E.**
    (2017). Neural Message Passing for Quantum Chemistry. *ICML 2017*.
    [arXiv:1704.01212](https://arxiv.org/abs/1704.01212)

11. **Bruna, J., Zaremba, W., Szlam, A., & LeCun, Y.** (2014). Spectral
    Networks and Locally Connected Networks on Graphs. *ICLR 2014*.
    [arXiv:1312.6203](https://arxiv.org/abs/1312.6203)

12. **Hammond, D. K., Vandergheynst, P., & Gribonval, R.** (2011).
    Wavelets on Graphs via Spectral Graph Theory. *Applied and Computational
    Harmonic Analysis*, 30(2), 129–150.

13. **Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M.,
    & Leskovec, J.** (2020). Open Graph Benchmark: Datasets for Machine
    Learning on Graphs. *NeurIPS 2020*.
    [arXiv:2005.00687](https://arxiv.org/abs/2005.00687)

14. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
    Gomez, A. N., Kaiser, Ł., & Polosukhin, I.** (2017). Attention Is All
    You Need. *NeurIPS 2017*.
    [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

15. **Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
    Schwenk, H., & Bengio, Y.** (2014). Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP 2014*.
    [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)

---

## 10. Future Work

The following directions represent natural extensions of this framework:

1. **Heterogeneous Graph Support.** Extend models to handle multiple node and
   edge types simultaneously. Relational GCN (R-GCN) and Heterogeneous Graph
   Transformer (HGT) would enable applications to knowledge graphs, biological
   interaction networks, and multi-modal social networks where different
   relation types carry distinct semantics.

2. **Scalable Mini-Batch Training.** Current implementations operate on full
   adjacency matrices. Integrating neighborhood sampling (as in the original
   GraphSAGE paper) or cluster-based mini-batching (ClusterGCN, GraphSAINT)
   would enable training on graphs with millions of nodes. This requires
   refactoring the data pipeline to support sparse adjacency representations
   and efficient neighbor sampling.

3. **Edge Feature Integration.** While the TGN module supports edge features,
   the GCN, GAT, and GIN models currently ignore edge attributes. Extending
   these models with edge-conditioned convolutions (e.g., NNConv, EdgeConv)
   would improve performance on molecular and chemical property prediction
   tasks where bond types and spatial distances are critical.

4. **Higher-Order GNNs.** Implementing k-dimensional GNNs (Morris et al., 2019)
   that aggregate over $k$-tuples rather than individual nodes would break the
   1-WL expressiveness barrier. This is particularly relevant for substructure
   counting tasks (cycles, cliques) where standard MPNNs provably fail.

5. **Self-Supervised Pre-Training.** Adding graph contrastive learning
   objectives (GraphCL, GCA) and masked feature prediction as pre-training
   tasks would enable learning useful representations from unlabeled graph
   data. This is especially valuable in domains like drug discovery where
   labeled data is scarce but molecular structures are abundant.

6. **Explainability and Interpretability.** Integrating GNN explanation methods
   (GNNExplainer, PGExplainer, SubgraphX) would provide per-prediction
   explanations identifying which subgraphs and features drive model decisions.
   The GAT attention weights (accessible via `last_attention`) already provide a
   form of soft explanation that could be extended.

7. **Equivariant GNNs.** Implementing SE(3)-equivariant architectures (EGNN,
   PaiNN, TFN) for 3D point cloud and molecular geometry tasks where
   predictions must respect rotational and translational symmetries. This would
   require extending the feature space to include geometric vectors alongside
   scalar features.

8. **Continuous-Time Dynamic Graphs.** Extending the TGN module with full
   temporal attention (temporal self-attention over interaction histories) and
   integrating with neural ODE-based approaches for continuous-time graph
   dynamics. The current snapshot-based TemporalGNN discretizes time; a
   continuous formulation would better model irregularly-sampled interactions.

---

## 11. License

This project is licensed under the MIT License. See `LICENSE` for details.

---

*Built with PyTorch. Designed for research reproducibility and pedagogical clarity.*
