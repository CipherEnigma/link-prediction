"""
Advanced Node Reputation Methods for Undirected Multilayer Networks

Implements:
1. Multiplex PageRank / MultiRank-style reputation
2. Eigenvector Versatility (supra-adjacency eigenvector centrality)
3. GNN-based Multiplex Reputation (PyTorch Geometric)

Author: ChatGPT
"""

# =========================
# Imports
# =========================
import math
import numpy as np
import networkx as nx

# Optional deep learning imports (only needed for GNN method)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# =========================
# SUPRA-GRAPH CONSTRUCTION
# =========================

def build_supra_graph_undirected(
    G_layers,
    interlayer_edges=None,
    w_intra=None,
    w_inter=None
):
    """
    Build an undirected supra-graph for a multilayer network.

    Parameters
    ----------
    G_layers : dict
        layer_id -> nx.Graph (undirected)
    interlayer_edges : list
        [((i, ell), (j, m)), ...]
    w_intra : dict
        layer_id -> intra-layer edge weight
    w_inter : dict
        frozenset({ell, m}) -> inter-layer weight
    """
    if interlayer_edges is None:
        interlayer_edges = []
    if w_intra is None:
        w_intra = {}
    if w_inter is None:
        w_inter = {}

    G_sup = nx.Graph()

    # Intra-layer edges
    for ell, G in G_layers.items():
        w = w_intra.get(ell, 1.0)
        for u in G.nodes():
            G_sup.add_node((u, ell))
        for u, v in G.edges():
            G_sup.add_edge((u, ell), (v, ell), weight=w)

    # Inter-layer edges
    for (u_ell, v_m) in interlayer_edges:
        (_, ell) = u_ell
        (_, m) = v_m
        w = w_inter.get(frozenset({ell, m}), 1.0)
        G_sup.add_edge(u_ell, v_m, weight=w)

    return G_sup

# =========================
# 1. MULTIPLEX PAGERANK
# =========================

def multiplex_pagerank(
    G_layers,
    interlayer_edges=None,
    w_intra=None,
    w_inter=None,
    alpha=0.85
):
    """
    Compute Multiplex PageRank on an undirected multilayer network.

    Returns
    -------
    R_supra : dict
        (node, layer) -> reputation
    R_node : dict
        node -> aggregated reputation across layers
    """
    G_sup = build_supra_graph_undirected(
        G_layers,
        interlayer_edges,
        w_intra,
        w_inter
    )

    R_supra = nx.pagerank(G_sup, alpha=alpha, weight="weight")

    R_node = {}
    for (i, ell), val in R_supra.items():
        R_node[i] = R_node.get(i, 0.0) + val

    return R_supra, R_node

# =========================
# 2. EIGENVECTOR VERSATILITY
# =========================

def eigenvector_versatility(
    G_layers,
    interlayer_edges=None,
    w_intra=None,
    w_inter=None,
    max_iter=300
):
    """
    Eigenvector versatility via principal eigenvector of supra-adjacency.
    """
    G_sup = build_supra_graph_undirected(
        G_layers,
        interlayer_edges,
        w_intra,
        w_inter
    )

    nodes = list(G_sup.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    A = np.zeros((n, n))
    for u, v, d in G_sup.edges(data=True):
        i, j = idx[u], idx[v]
        w = d.get("weight", 1.0)
        A[i, j] += w
        A[j, i] += w

    # Power iteration
    x = np.random.rand(n)
    x /= np.linalg.norm(x)

    for _ in range(max_iter):
        x_new = A @ x
        if np.linalg.norm(x_new) == 0:
            break
        x_new /= np.linalg.norm(x_new)
        if np.linalg.norm(x_new - x) < 1e-10:
            break
        x = x_new

    c_supra = {nodes[i]: float(x[i]) for i in range(n)}

    V_node = {}
    for (i, ell), val in c_supra.items():
        V_node[i] = V_node.get(i, 0.0) + val

    return c_supra, V_node

# =========================
# 3. GNN-BASED MULTIPLEX REPUTATION
# =========================

if TORCH_AVAILABLE:

    class MultiplexReputationGNN(nn.Module):
        """
        GNN-based reputation for undirected multilayer networks.
        """
        def __init__(self, num_layers, in_dim, hidden_dim=64):
            super().__init__()
            self.num_layers = num_layers
            self.gcns = nn.ModuleList([
                GCNConv(in_dim, hidden_dim)
                for _ in range(num_layers)
            ])
            self.att = nn.Parameter(torch.randn(hidden_dim))
            self.out = nn.Linear(hidden_dim, 1)

        def forward(self, x, edge_index_per_layer):
            layer_embeds = []
            att_scores = []

            for ell in range(self.num_layers):
                h = F.relu(self.gcns[ell](x, edge_index_per_layer[ell]))
                layer_embeds.append(h)
                att_scores.append((h * self.att).sum(dim=1))

            A = torch.stack(att_scores, dim=0)
            alpha = F.softmax(A, dim=0)

            H = torch.stack(layer_embeds, dim=0)
            z = (alpha.unsqueeze(-1) * H).sum(dim=0)

            reputation = torch.sigmoid(self.out(z)).squeeze()
            return reputation

# =========================
# EXAMPLE USAGE
# =========================
if __name__ == "__main__":
    # Simple example with 2 layers
    G1 = nx.Graph()
    G1.add_edges_from([(0, 1), (1, 2)])

    G2 = nx.Graph()
    G2.add_edges_from([(0, 2), (2, 3)])

    G_layers = {0: G1, 1: G2}

    inter_edges = [((i, 0), (i, 1)) for i in G1.nodes()]

    print("=== Multiplex PageRank ===", flush=True)
    R_supra, R_node = multiplex_pagerank(G_layers, inter_edges)
    print(R_node, flush=True)

    print("\n=== Eigenvector Versatility ===", flush=True)
    c_supra, V_node = eigenvector_versatility(G_layers, inter_edges)
    print(V_node, flush=True)

    if TORCH_AVAILABLE:
        print("\nGNN model is available for training.", flush=True)
    else:
        print("\nPyTorch Geometric not installed; GNN method skipped.", flush=True)
