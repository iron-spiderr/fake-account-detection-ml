"""
Module 3 — Graph Construction
Builds a social-network interaction graph and computes per-node centrality features.
Optionally trains a GCN+GAT for 64-d node embeddings (requires torch_geometric).
"""

import logging

import numpy as np
import pandas as pd
import networkx as nx

logger = logging.getLogger(__name__)

GRAPH_FEATURE_COLS = [
    "pagerank", "betweenness_centrality",
    "clustering_coefficient", "degree_centrality",
    "community_size_ratio",
]


# ---------------------------------------------------------------------------
# 3.1  Build interaction graph
# ---------------------------------------------------------------------------

def build_interaction_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Build an undirected graph where each node is a user (row index).
    Edges are added between users that share follower/following similarity
    (simple proxy: bucket by follower count tier, connecting peers).
    """
    G = nx.Graph()
    G.add_nodes_from(df.index.tolist())

    followers = pd.to_numeric(df.get("followers", pd.Series([0] * len(df),
                                                              index=df.index)),
                               errors="coerce").fillna(0)

    # Bucket users into log-scale tiers, connect within tiers
    log_f = np.log1p(followers.values)
    bins = np.linspace(log_f.min(), log_f.max() + 1e-9, 20)
    tiers = np.digitize(log_f, bins)

    idx = df.index.tolist()
    tier_map: dict[int, list] = {}
    for pos, node in enumerate(idx):
        t = int(tiers[pos])
        tier_map.setdefault(t, []).append(node)

    edge_count = 0
    MAX_EDGES_PER_TIER = 200
    rng = np.random.default_rng(42)
    for members in tier_map.values():
        if len(members) > 1:
            # Sample edges to keep graph sparse
            pairs = [(members[i], members[j])
                     for i in range(len(members))
                     for j in range(i + 1, min(i + 10, len(members)))]
            sampled = rng.choice(len(pairs), min(MAX_EDGES_PER_TIER, len(pairs)),
                                  replace=False)
            for s in sampled:
                G.add_edge(*pairs[s])
                edge_count += 1

    logger.info("Graph: %d nodes, %d edges", G.number_of_nodes(), edge_count)
    return G


# ---------------------------------------------------------------------------
# 3.2  Centrality features
# ---------------------------------------------------------------------------

def compute_centrality_features(G: nx.Graph, index: pd.Index) -> pd.DataFrame:
    """
    Compute 5 centrality metrics per node.
    Returns a DataFrame aligned to `index`.
    """
    nodes = list(index)

    # PageRank
    pr = nx.pagerank(G, max_iter=100, tol=1e-6)

    # Betweenness (approximate for large graphs)
    n = G.number_of_nodes()
    k = min(100, n) if n > 200 else None
    btw = nx.betweenness_centrality(G, k=k, normalized=True, seed=42)

    # Clustering coefficient
    clust = nx.clustering(G)

    # Degree centrality
    deg = nx.degree_centrality(G)

    # Community size ratio via Louvain (greedy modular if louvain unavailable)
    try:
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(G, seed=42)
    except Exception:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))

    node_community_size: dict = {}
    total = G.number_of_nodes()
    for comm in communities:
        ratio = len(comm) / total
        for node in comm:
            node_community_size[node] = ratio

    data = {
        "pagerank": [pr.get(n, 0.0) for n in nodes],
        "betweenness_centrality": [btw.get(n, 0.0) for n in nodes],
        "clustering_coefficient": [clust.get(n, 0.0) for n in nodes],
        "degree_centrality": [deg.get(n, 0.0) for n in nodes],
        "community_size_ratio": [node_community_size.get(n, 0.0) for n in nodes],
    }
    return pd.DataFrame(data, index=index)


# ---------------------------------------------------------------------------
# Optional GNN embeddings (requires torch_geometric)
# ---------------------------------------------------------------------------

def compute_gnn_embeddings(G: nx.Graph, labels: pd.Series,
                            index: pd.Index, emb_dim: int = 64) -> pd.DataFrame | None:
    """Train a simple GCN and return node embeddings. Returns None on failure."""
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv, GATConv
    except ImportError:
        logger.info("torch_geometric not found; skipping GNN embeddings.")
        return None

    nodes = list(index)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Build edge index
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()
             if u in node_to_idx and v in node_to_idx]
    if not edges:
        return None

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.tensor(labels.reindex(index).fillna(0).values, dtype=torch.long)
    num_nodes = len(nodes)
    x = torch.eye(num_nodes, dtype=torch.float)  # identity node features

    data = Data(x=x, edge_index=edge_index, y=y)

    class GNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(num_nodes, 128)
            self.conv2 = GATConv(128, emb_dim, heads=1)

        def forward(self, data):
            x = F.relu(self.conv1(data.x, data.edge_index))
            x = self.conv2(x, data.edge_index)
            return x

    model = GNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        emb = model(data)
        loss = F.cross_entropy(emb, y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        emb = model(data).numpy()

    cols = [f"gnn_{i}" for i in range(emb_dim)]
    return pd.DataFrame(emb, index=index, columns=cols)


# ---------------------------------------------------------------------------
# 3.3  Main entry point
# ---------------------------------------------------------------------------

def build_graph_features(df: pd.DataFrame,
                          labels: pd.Series | None = None,
                          use_gnn: bool = False) -> pd.DataFrame:
    """
    Build graph + centrality features for all rows in `df`.
    Optionally append GNN embeddings.
    Returns a DataFrame aligned to df.index.
    """
    logger.info("=== Module 3: Graph Construction ===")
    G = build_interaction_graph(df)
    feat = compute_centrality_features(G, df.index)

    if use_gnn and labels is not None:
        gnn_emb = compute_gnn_embeddings(G, labels, df.index)
        if gnn_emb is not None:
            feat = pd.concat([feat, gnn_emb], axis=1)

    return feat
