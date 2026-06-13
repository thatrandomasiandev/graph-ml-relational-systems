"""Graph Convolutional Network, Graph Attention Network, and Graph Isomorphism Network."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_ml.models.layers import normalized_adjacency, propagate

logger = logging.getLogger(__name__)


class GCN(nn.Module):
    """Multi-layer GCN with ReLU activations and dropout.

    Implements the spectral convolution of Kipf & Welling (2017):
        H^{(l+1)} = σ(D̃^{-1/2} Ã D̃^{-1/2} H^{(l)} W^{(l)})

    Args:
        in_dim: Dimensionality of input node features.
        hidden_dim: Width of hidden layers.
        out_dim: Dimensionality of output node embeddings.
        n_layers: Number of message-passing layers (must be >= 1).
        dropout: Dropout probability applied between layers.
    """

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

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass through all GCN layers.

        Args:
            x: Node feature matrix of shape ``(N, in_dim)``.
            adj: Adjacency matrix of shape ``(N, N)``.

        Returns:
            Node embeddings of shape ``(N, out_dim)``.
        """
        adj_norm = normalized_adjacency(adj)
        h = x
        for i, layer in enumerate(self.weights):
            h = propagate(adj_norm, h)
            h = layer(h)
            if i < self.n_layers - 1:
                h = torch.relu(h)
                h = self.dropout(h)
        return h


# ---------------------------------------------------------------------------
# Graph Attention Network (Veličković et al., 2018)
# ---------------------------------------------------------------------------


class _GATHead(nn.Module):
    """Single attention head for GAT.

    Computes attention coefficients:
        e_{ij} = LeakyReLU(a^T [Wh_i || Wh_j])
        α_{ij} = softmax_j(e_{ij})
        h'_i   = σ(Σ_j α_{ij} W h_j)

    Args:
        in_dim: Input feature dimensionality.
        out_dim: Output dimensionality for this head.
        dropout: Dropout applied to attention coefficients.
        alpha: Negative slope for LeakyReLU.
    """

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

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(
        self, h: torch.Tensor, adj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute one attention head.

        Args:
            h: Node features ``(N, in_dim)``.
            adj: Adjacency matrix ``(N, N)`` (entries > 0 indicate edges).

        Returns:
            Tuple of (output features ``(N, out_dim)``, attention weights ``(N, N)``).
        """
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


class GraphAttentionNetwork(nn.Module):
    """Multi-head Graph Attention Network (GAT).

    Each layer applies *K* independent attention heads; intermediate layers
    concatenate head outputs while the final layer averages them.

    Math per head *k*:
        e_{ij}^k  = LeakyReLU(a_k^T [W_k h_i || W_k h_j])
        α_{ij}^k  = softmax_j(e_{ij}^k)
        h_i'^k    = ELU(Σ_j α_{ij}^k W_k h_j)

    Args:
        in_dim: Input feature dimensionality.
        hidden_dim: Per-head hidden dimensionality.
        out_dim: Final output dimensionality.
        n_heads: Number of attention heads per layer.
        dropout: Dropout probability for attention coefficients and features.
        alpha: Negative slope for LeakyReLU in attention.

    Returns:
        Node embeddings of shape ``(N, out_dim)`` from :meth:`forward`.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.2,
        alpha: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.dropout = nn.Dropout(p=dropout)

        self.heads = nn.ModuleList([
            _GATHead(in_dim, hidden_dim, dropout=dropout, alpha=alpha)
            for _ in range(n_heads)
        ])
        self.out_proj = nn.Linear(hidden_dim * n_heads, out_dim)
        self._last_attention: torch.Tensor | None = None

    @property
    def last_attention(self) -> torch.Tensor | None:
        """Attention weights from the most recent forward pass ``(n_heads, N, N)``."""
        return self._last_attention

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-head attention and output projection.

        Args:
            x: Node features ``(N, in_dim)``.
            adj: Adjacency matrix ``(N, N)``.

        Returns:
            Node embeddings ``(N, out_dim)``.
        """
        x = self.dropout(x)
        head_outs: list[torch.Tensor] = []
        attns: list[torch.Tensor] = []
        for head in self.heads:
            out, attn = head(x, adj)
            head_outs.append(F.elu(out))
            attns.append(attn)

        self._last_attention = torch.stack(attns, dim=0)
        h = torch.cat(head_outs, dim=-1)  # (N, n_heads * hidden_dim)
        h = self.out_proj(h)
        return h


# ---------------------------------------------------------------------------
# Graph Isomorphism Network (Xu et al., 2019)
# ---------------------------------------------------------------------------


class _GINLayer(nn.Module):
    """Single GIN layer with learnable epsilon.

    Implements:
        h_v^{(k)} = MLP^{(k)}((1 + ε^{(k)}) · h_v^{(k-1)} + Σ_{u ∈ N(v)} h_u^{(k-1)})

    Args:
        in_dim: Input feature dimensionality.
        hidden_dim: MLP hidden width.
        out_dim: Output dimensionality.
    """

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
        """Apply GIN update rule.

        Args:
            h: Node features ``(N, in_dim)``.
            adj: Adjacency matrix ``(N, N)`` (no self-loops expected).

        Returns:
            Updated features ``(N, out_dim)``.
        """
        neigh_sum = adj @ h
        out = (1.0 + self.eps) * h + neigh_sum
        return self.mlp(out)


class GraphIsomorphismNetwork(nn.Module):
    """Graph Isomorphism Network for graph-level classification.

    Stacks *K* GIN layers and applies sum-pooling READOUT for graph-level
    representations. Node-level embeddings are also available.

    Math:
        h_v^{(k)} = MLP^{(k)}((1 + ε^{(k)}) · h_v^{(k-1)} + Σ_{u ∈ N(v)} h_u^{(k-1)})
        h_G = Σ_{k=0}^{K} READOUT({h_v^{(k)} | v ∈ G})

    where READOUT is sum pooling.

    Args:
        in_dim: Input feature dimensionality.
        hidden_dim: Hidden layer width.
        out_dim: Final output dimensionality.
        n_layers: Number of GIN layers (must be >= 1).
        dropout: Dropout probability between layers.

    Returns:
        Node embeddings ``(N, out_dim)`` from :meth:`forward`, or
        graph-level embedding ``(out_dim,)`` from :meth:`graph_forward`.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)

        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            _GINLayer(dims[i], hidden_dim, dims[i + 1])
            for i in range(n_layers)
        ])
        self.initial_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Node-level forward pass.

        Args:
            x: Node features ``(N, in_dim)``.
            adj: Adjacency matrix ``(N, N)`` (no self-loops required).

        Returns:
            Node embeddings ``(N, out_dim)``.
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, adj)
            if i < self.n_layers - 1:
                h = self.dropout(h)
        return h

    def graph_forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Graph-level forward with sum-pooling READOUT across all layers.

        Concatenates sum-pooled representations from every layer (including
        the initial projection of layer 0) for maximum expressiveness.

        Args:
            x: Node features ``(N, in_dim)``.
            adj: Adjacency matrix ``(N, N)``.
            batch: Graph membership indices ``(N,)`` for batched graphs.
                If ``None``, all nodes belong to a single graph.

        Returns:
            Graph-level embedding ``(n_graphs, out_dim * (n_layers + 1))``.
        """
        layer_readouts: list[torch.Tensor] = []
        h = x
        layer_readouts.append(self._readout(self.initial_proj(h), batch))

        for i, layer in enumerate(self.layers):
            h = layer(h, adj)
            if i < self.n_layers - 1:
                h = self.dropout(h)
            layer_readouts.append(self._readout(h, batch))

        return torch.cat(layer_readouts, dim=-1)

    @staticmethod
    def _readout(h: torch.Tensor, batch: torch.Tensor | None) -> torch.Tensor:
        """Sum pooling over nodes, respecting batch assignment.

        Args:
            h: Node features ``(N, D)``.
            batch: Graph indices ``(N,)`` or ``None``.

        Returns:
            Graph-level features ``(n_graphs, D)``.
        """
        if batch is None:
            return h.sum(dim=0, keepdim=True)
        n_graphs = int(batch.max().item()) + 1
        out = torch.zeros(n_graphs, h.shape[-1], device=h.device, dtype=h.dtype)
        return out.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)
