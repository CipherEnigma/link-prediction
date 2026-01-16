# factorization_machine.py
"""
Factorization Machine (FM) for link prediction.
- Featureization: for node pair (i,j) we create x = [onehot(i); onehot(j)] (size 2n)
- Model computes: y_hat = w0 + w^T x + 0.5 * sum_f ( (xV)_f^2 - (x^2 (V^2))_f )
- Trains with BCE on sampled positives/negatives
- Note: this is a demonstration; for larger graphs use sparse features or hashed features
"""
import argparse, random, numpy as np
import torch, torch.nn as nn, torch.optim as optim
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def train_test_split_edges(adj, test_frac=0.2, seed=42):
    np.random.seed(seed)
    upper = np.triu(np.ones_like(adj), k=1).astype(bool)
    edges = np.array(np.where((adj > 0) & upper)).T
    non_edges = np.array(np.where((adj == 0) & upper)).T
    idx = np.random.permutation(len(edges))
    split = int(len(edges) * (1 - test_frac))
    train_edges = edges[idx[:split]]
    test_edges = edges[idx[split:]]
    return train_edges, test_edges, non_edges

def sample_negatives(non_edges, n_samples, seed=42):
    np.random.seed(seed)
    idx = np.random.choice(len(non_edges), size=n_samples, replace=False)
    return non_edges[idx]

class FM(nn.Module):
    def __init__(self, n_features, k):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.zeros(n_features))
        # latent factors V (n_features x k)
        self.V = nn.Parameter(torch.randn(n_features, k) * 0.01)

    def forward(self, x):  # x: (batch, n_features)
        linear = x @ self.w  # (batch,)
        # efficient pairwise interaction
        # sum1 = (x @ V) ^ 2
        sum1 = (x @ self.V) ** 2  # (batch, k)
        sum2 = ( (x ** 2) @ (self.V ** 2) )  # (batch, k)
        interaction = 0.5 * torch.sum(sum1 - sum2, dim=1)  # (batch,)
        return self.w0 + linear + interaction

def make_feature_vector(n_nodes, pair):
    i, j = pair
    vec = np.zeros(2 * n_nodes, dtype=np.float32)
    vec[i] = 1.0
    vec[n_nodes + j] = 1.0
    return vec

def evaluate_model(model, device, test_edges, non_edges, n_nodes, seed=42, num_neg=None):
    if num_neg is None: num_neg = len(test_edges)
    negs = sample_negatives(non_edges, num_neg, seed)
    X = []
    y = []
    for (i,j) in test_edges:
        X.append(make_feature_vector(n_nodes, (i,j)))
        y.append(1)
    for (i,j) in negs:
        X.append(make_feature_vector(n_nodes, (i,j)))
        y.append(0)
    X = torch.tensor(np.stack(X), device=device)
    with torch.no_grad():
        logits = model(X).cpu().numpy()
    y = np.array(y)
    return roc_auc_score(y, logits), average_precision_score(y, logits)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=8)  # FM latent dim
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    G = nx.karate_club_graph()
    n_nodes = G.number_of_nodes()
    A = nx.to_numpy_array(G)

    train_edges, test_edges, non_edges = train_test_split_edges(A, test_frac=0.2, seed=args.seed)
    print(f"Nodes: {n_nodes}, train edges: {len(train_edges)}, test edges: {len(test_edges)}", flush=True)

    n_features = 2 * n_nodes
    model = FM(n_features, args.k).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    train_pairs = train_edges.copy()
    for epoch in range(1, args.epochs + 1):
        # minibatch
        perm = np.random.permutation(len(train_pairs))
        batch_size = min(256, len(train_pairs))
        batch_idx = perm[:batch_size]
        pos_batch = train_pairs[batch_idx]
        neg_batch = sample_negatives(non_edges, batch_size, seed=args.seed + epoch)

        X_pos = np.stack([make_feature_vector(n_nodes, (i,j)) for i,j in pos_batch])
        X_neg = np.stack([make_feature_vector(n_nodes, (i,j)) for i,j in neg_batch])
        X = torch.tensor(np.vstack([X_pos, X_neg]), device=device)
        y = torch.tensor(np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))]), device=device)

        logits = model(X)
        loss = bce(logits, y)

        opt.zero_grad(); loss.backward(); opt.step()

        if epoch % 200 == 0 or epoch == 1:
            auc, ap = evaluate_model(model, device, test_edges, non_edges, n_nodes, seed=args.seed)
            print(f"Epoch {epoch}/{args.epochs} loss={loss.item():.4f} AUC={auc:.4f} AP={ap:.4f}", flush=True)

    auc, ap = evaluate_model(model, device, test_edges, non_edges, n_nodes, seed=args.seed)
    print("Final FM -> AUC: {:.4f}, AP: {:.4f}".format(auc, ap), flush=True)

if __name__ == "__main__":
    main()
