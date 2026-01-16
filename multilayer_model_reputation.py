"""
node_reputation_models.py

Comprehensive node reputation models for undirected social
and multilayer (multiplex) networks.

Implemented models:
1. Eigenvector / EigenTrust-style reputation
2. PageRank reputation
3. Random Walk with Restart (RWR)
4. Bayesian (Beta) reputation model
5. Multiplex PageRank (multilayer)
6. Multilayer Random Walk with Restart

Dependencies:
    pip install numpy networkx
"""

import numpy as np
import networkx as nx

# --------------------------------------------------
# Utility Functions
# --------------------------------------------------

def normalize(v):
    s = np.sum(v)
    return v if s == 0 else v / s


def row_normalize(M):
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return M / row_sums


# --------------------------------------------------
# 1. EigenTrust / Eigenvector Reputation
# --------------------------------------------------

def eigentrust_reputation(G, alpha=0.15, max_iter=100, tol=1e-6):
    """
    EigenTrust-style global reputation

    t = (1 - alpha) * C^T t + alpha * p
    """
    A = nx.to_numpy_array(G)
    C = row_normalize(A)
    n = C.shape[0]

    t = np.ones(n) / n
    p = t.copy()

    for _ in range(max_iter):
        t_new = (1 - alpha) * C.T @ t + alpha * p
        if np.linalg.norm(t_new - t, 1) < tol:
            break
        t = t_new

    return normalize(t)


# --------------------------------------------------
# 2. PageRank Reputation
# --------------------------------------------------

def pagerank_reputation(G, alpha=0.85):
    """
    PageRank as node reputation
    """
    return nx.pagerank(G, alpha=alpha)


# --------------------------------------------------
# 3. Random Walk with Restart (RWR)
# --------------------------------------------------

def random_walk_with_restart(G, seed_node, c=0.85, max_iter=100):
    """
    Personalized reputation centered on seed_node
    """
    nodes = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes)
    P = row_normalize(A)

    n = len(nodes)
    e = np.zeros(n)
    e[nodes.index(seed_node)] = 1

    r = e.copy()
    for _ in range(max_iter):
        r = (1 - c) * e + c * P.T @ r

    return dict(zip(nodes, normalize(r)))


# --------------------------------------------------
# 4. Bayesian (Beta) Reputation Model
# --------------------------------------------------

class BayesianReputation:
    """
    Bayesian reputation using Beta distribution
    """

    def __init__(self, alpha=1, beta=1):
        self.alpha = alpha
        self.beta = beta

    def update(self, positive=0, negative=0):
        self.alpha += positive
        self.beta += negative

    def reputation(self):
        """
        Expected value of Beta distribution
        """
        return self.alpha / (self.alpha + self.beta)


# --------------------------------------------------
# 5. Multiplex PageRank (Multilayer Networks)
# --------------------------------------------------

def multiplex_pagerank(layers, omega=1.0, alpha=0.85, max_iter=100):
    """
    Multiplex PageRank using supra-adjacency matrix
    """
    nodes = list(layers[0].nodes())
    n = len(nodes)
    L = len(layers)

    S = np.zeros((n * L, n * L))

    # Intra-layer edges
    for l, G in enumerate(layers):
        A = nx.to_numpy_array(G, nodelist=nodes)
        S[l*n:(l+1)*n, l*n:(l+1)*n] = A

    # Inter-layer coupling
    for l in range(L):
        for m in range(L):
            if l != m:
                S[l*n:(l+1)*n, m*n:(m+1)*n] = omega * np.eye(n)

    P = row_normalize(S)

    r = np.ones(n * L) / (n * L)
    v = r.copy()

    for _ in range(max_iter):
        r = alpha * P.T @ r + (1 - alpha) * v

    r = r.reshape(L, n).sum(axis=0)
    return dict(zip(nodes, normalize(r)))


# --------------------------------------------------
# 6. Multilayer Random Walk with Restart
# --------------------------------------------------

def multilayer_rwr(layers, seed_node, omega=1.0, c=0.85, max_iter=100):
    """
    Multilayer Random Walk with Restart
    """
    nodes = list(layers[0].nodes())
    n = len(nodes)
    L = len(layers)

    S = np.zeros((n * L, n * L))

    for l, G in enumerate(layers):
        A = nx.to_numpy_array(G, nodelist=nodes)
        S[l*n:(l+1)*n, l*n:(l+1)*n] = A

    for l in range(L):
        for m in range(L):
            if l != m:
                S[l*n:(l+1)*n, m*n:(m+1)*n] = omega * np.eye(n)

    P = row_normalize(S)

    e = np.zeros(n * L)
    idx = nodes.index(seed_node)
    for l in range(L):
        e[l*n + idx] = 1 / L

    r = e.copy()
    for _ in range(max_iter):
        r = (1 - c) * e + c * P.T @ r

    r = r.reshape(L, n).sum(axis=0)
    return dict(zip(nodes, normalize(r)))


# --------------------------------------------------
# Analysis and Visualization
# --------------------------------------------------

def print_top_nodes(reputation_dict, model_name, top_k=5):
    """Print top-k nodes by reputation score"""
    sorted_nodes = sorted(reputation_dict.items(), 
                         key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*60}", flush=True)
    print(f"{model_name} - Top {top_k} Nodes", flush=True)
    print(f"{'='*60}", flush=True)
    
    for rank, (node, score) in enumerate(sorted_nodes[:top_k], 1):
        print(f"  Rank {rank}: Node {node:>3} | Score: {score:.6f}", flush=True)


def analyze_reputation_stats(reputation_dict, model_name):
    """Print statistical summary of reputation scores"""
    scores = np.array(list(reputation_dict.values()))
    
    print(f"\n{'-'*60}", flush=True)
    print(f"{model_name} - Statistical Summary", flush=True)
    print(f"{'-'*60}", flush=True)
    print(f"  Mean:        {np.mean(scores):.6f}", flush=True)
    print(f"  Std Dev:     {np.std(scores):.6f}", flush=True)
    print(f"  Min:         {np.min(scores):.6f}", flush=True)
    print(f"  Max:         {np.max(scores):.6f}", flush=True)
    print(f"  Median:      {np.median(scores):.6f}", flush=True)


def compare_models(results_dict):
    """Compare which nodes are ranked highest across different models"""
    print(f"\n{'='*60}", flush=True)
    print("MODEL COMPARISON - Top Node Across All Models", flush=True)
    print(f"{'='*60}", flush=True)
    
    for model_name, reputation in results_dict.items():
        top_node = max(reputation.items(), key=lambda x: x[1])
        print(f"  {model_name:<25} → Node {top_node[0]:>3} (score: {top_node[1]:.6f})", flush=True)


# --------------------------------------------------
# Example Usage with Detailed Analysis
# --------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60, flush=True)
    print("NODE REPUTATION ANALYSIS - Karate Club Network", flush=True)
    print("="*60, flush=True)
    
    # Load Karate Club graph
    G = nx.karate_club_graph()
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    print(f"\nNetwork Statistics:", flush=True)
    print(f"  Nodes: {n_nodes}", flush=True)
    print(f"  Edges: {n_edges}", flush=True)
    print(f"  Avg Degree: {2*n_edges/n_nodes:.2f}", flush=True)
    
    # Store all results
    results = {}
    
    # 1. EigenTrust
    print("\n\n" + "#"*60, flush=True)
    print("# 1. EIGENTRUST REPUTATION", flush=True)
    print("#"*60, flush=True)
    print("\nEigenTrust measures global trust by iteratively propagating", flush=True)
    print("trust scores through the network based on peer recommendations.", flush=True)
    
    eigentrust_scores = eigentrust_reputation(G)
    results['EigenTrust'] = dict(enumerate(eigentrust_scores))
    print_top_nodes(results['EigenTrust'], "EigenTrust")
    analyze_reputation_stats(results['EigenTrust'], "EigenTrust")
    
    # 2. PageRank
    print("\n\n" + "#"*60, flush=True)
    print("# 2. PAGERANK REPUTATION", flush=True)
    print("#"*60, flush=True)
    print("\nPageRank treats connections as endorsements. Nodes connected", flush=True)
    print("to important nodes receive higher reputation scores.", flush=True)
    
    pagerank_scores = pagerank_reputation(G)
    results['PageRank'] = pagerank_scores
    print_top_nodes(results['PageRank'], "PageRank")
    analyze_reputation_stats(results['PageRank'], "PageRank")
    
    # 3. Random Walk with Restart
    print("\n\n" + "#"*60, flush=True)
    print("# 3. RANDOM WALK WITH RESTART (Personalized)", flush=True)
    print("#"*60, flush=True)
    print("\nRWR computes personalized reputation from the perspective of", flush=True)
    print("a seed node (Node 0 in this case). Scores reflect proximity", flush=True)
    print("and importance relative to the seed.", flush=True)
    
    rwr_scores = random_walk_with_restart(G, 0)
    results['RWR (seed=0)'] = rwr_scores
    print_top_nodes(results['RWR (seed=0)'], "Random Walk with Restart")
    analyze_reputation_stats(results['RWR (seed=0)'], "RWR")
    
    # 4. Bayesian Reputation
    print("\n\n" + "#"*60, flush=True)
    print("# 4. BAYESIAN REPUTATION (Single Node Example)", flush=True)
    print("#"*60, flush=True)
    print("\nBayesian model uses Beta distribution to model reputation", flush=True)
    print("based on positive and negative feedback counts.", flush=True)
    
    br = BayesianReputation()
    br.update(positive=10, negative=3)
    bayesian_score = br.reputation()
    print(f"\nExample: 10 positive, 3 negative interactions", flush=True)
    print(f"  Reputation Score: {bayesian_score:.6f}", flush=True)
    print(f"  Interpretation: {bayesian_score*100:.2f}% trustworthiness", flush=True)
    
    # 5. Multiplex PageRank
    print("\n\n" + "#"*60, flush=True)
    print("# 5. MULTIPLEX PAGERANK (Multilayer)", flush=True)
    print("#"*60, flush=True)
    print("\nMultiplex PageRank aggregates reputation across multiple", flush=True)
    print("network layers (e.g., friendship + collaboration networks).", flush=True)
    
    # Create second layer
    G2 = nx.erdos_renyi_graph(G.number_of_nodes(), 0.1, seed=42)
    print(f"\nLayer 1 (Karate): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)
    print(f"Layer 2 (Random): {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges", flush=True)
    
    multiplex_scores = multiplex_pagerank([G, G2])
    results['Multiplex PageRank'] = multiplex_scores
    print_top_nodes(results['Multiplex PageRank'], "Multiplex PageRank")
    analyze_reputation_stats(results['Multiplex PageRank'], "Multiplex PageRank")
    
    # 6. Multilayer RWR
    print("\n\n" + "#"*60, flush=True)
    print("# 6. MULTILAYER RANDOM WALK WITH RESTART", flush=True)
    print("#"*60, flush=True)
    print("\nMultilayer RWR extends personalized reputation across multiple", flush=True)
    print("network layers, allowing random walks to jump between layers.", flush=True)
    
    multilayer_rwr_scores = multilayer_rwr([G, G2], seed_node=0)
    results['Multilayer RWR'] = multilayer_rwr_scores
    print_top_nodes(results['Multilayer RWR'], "Multilayer RWR")
    analyze_reputation_stats(results['Multilayer RWR'], "Multilayer RWR")
    
    # Final Comparison
    print("\n\n" + "="*60, flush=True)
    print("FINAL COMPARISON", flush=True)
    print("="*60, flush=True)
    compare_models(results)
    
    print("\n" + "="*60, flush=True)
    print("KEY INSIGHTS", flush=True)
    print("="*60, flush=True)
    print("\n1. EigenTrust & PageRank identify globally central nodes", flush=True)
    print("2. RWR methods are personalized (bias toward seed node)", flush=True)
    print("3. Multilayer methods capture cross-layer importance", flush=True)
    print("4. Bayesian method is local (per-node feedback based)", flush=True)
    print("\n" + "="*60 + "\n", flush=True)
