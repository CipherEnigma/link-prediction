# graph_nmf_link_prediction.py
"""
Graph NMF via multiplicative updates with Laplacian regularization.
- Factorizes A_train ≈ W W^T with W >= 0
- Update rule: W *= (A @ W) / (W @ (W.T @ W) + alpha * (L @ W) + eps)
- Evaluates using AUC/AP on held-out edges
"""
import argparse, numpy as np
from scipy.sparse import csgraph
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx

def train_test_split_edges(adj, test_frac=0.2, seed=42):
    np.random.seed(seed)
    upper = np.triu(np.ones_like(adj), k=1).astype(bool)
    edges = np.array(np.where((adj > 0) & upper)).T
    non_edges = np.array(np.where((adj == 0) & upper)).T
    idx = np.random.permutation(len(edges))
    split = int(len(edges) * (1 - test_frac))
    train_edges = edges[idx[:split]]
    test_edges = edges[idx[split:]]
    train_adj = np.zeros_like(adj)
    for i,j in train_edges:
        train_adj[i,j] = train_adj[j,i] = 1
    return train_adj, train_edges, test_edges, non_edges

def sample_negatives(non_edges, n_samples, seed=42):
    np.random.seed(seed)
    idx = np.random.choice(len(non_edges), size=n_samples, replace=False)
    return non_edges[idx]

def evaluate_scores(scores, test_edges, non_edges, num_neg=None, seed=42):
    if num_neg is None: num_neg = len(test_edges)
    negs = sample_negatives(non_edges, num_neg, seed)
    pos_scores = [scores[i,j] for (i,j) in test_edges]
    neg_scores = [scores[i,j] for (i,j) in negs]
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.concatenate([pos_scores, neg_scores])
    return roc_auc_score(y_true, y_scores), average_precision_score(y_true, y_scores)

def graph_nmf(A_train, k=8, alpha=0.1, max_iter=500, tol=1e-4):
    n = A_train.shape[0]
    W = np.maximum(0.1 * np.random.rand(n, k), 1e-8)
    L = csgraph.laplacian(A_train, normed=False)  # dense numpy array
    eps = 1e-8

    for it in range(max_iter):
        W_old = W.copy()
        numerator = A_train @ W  # (n,k)
        denom = W @ (W.T @ W) + alpha * (L @ W) + eps  # (n,k)
        W *= numerator / denom
        # convergence check
        diff = np.linalg.norm(W - W_old, ord='fro')
        if (it+1) % 50 == 0 or it == 0:
            print(f"Iter {it+1}/{max_iter}, ||W-W_old||_F = {diff:.6f}")
        if diff < tol:
            print(f"Converged at iter {it+1}")
            break
    return W

def main():
    import sys
    print("Script started!", flush=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--max_iter', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print(f"Arguments: k={args.k}, alpha={args.alpha}, max_iter={args.max_iter}, seed={args.seed}", flush=True)

    np.random.seed(args.seed)

    print("Loading Karate Club graph...", flush=True)
    G = nx.karate_club_graph()
    A = nx.to_numpy_array(G)
    A_train, train_edges, test_edges, non_edges = train_test_split_edges(A, test_frac=0.2, seed=args.seed)
    print(f"Nodes: {A.shape[0]}, train edges: {len(train_edges)}, test edges: {len(test_edges)}", flush=True)

    print("Starting Graph NMF training...", flush=True)
    W = graph_nmf(A_train, k=args.k, alpha=args.alpha, max_iter=args.max_iter)
    A_pred = W @ W.T  # reconstructed scores (non-negative)
    # normalize to [0,1]
    A_pred_norm = (A_pred - A_pred.min()) / (A_pred.max() - A_pred.min() + 1e-12)

    print("Evaluating...", flush=True)
    auc, ap = evaluate_scores(A_pred_norm, test_edges, non_edges, seed=args.seed)
    print("Graph NMF final -> AUC: {:.4f}, AP: {:.4f}".format(auc, ap), flush=True)

if __name__ == "__main__":
    main()
