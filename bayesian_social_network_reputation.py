"""
Bayesian Node Reputation Model for Undirected Social Networks
------------------------------------------------------------

This script computes user/node reputation in an undirected social network
using a Bayesian Beta-Bernoulli model with network-weighted feedback.

Author: (You can add your name)
Date: 2025

Requirements:
    numpy
    networkx

Install via:
    pip install numpy networkx
"""

import numpy as np
import networkx as nx


# -------------------------------------------------------
# 1. Generate an undirected social network
# -------------------------------------------------------

def generate_social_network(num_nodes=100, edge_prob=0.05, seed=42):
    """
    Generates a connected undirected social network.
    """
    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(num_nodes, edge_prob, seed=seed)

    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(num_nodes, edge_prob, seed=int(rng.integers(1e6)))

    return G


# -------------------------------------------------------
# 2. Generate ground-truth user behavior (theta)
# -------------------------------------------------------

def generate_true_reputation(num_nodes, alpha=4.0, beta=2.0, seed=42):
    """
    Generates latent user trustworthiness values.
    """
    rng = np.random.default_rng(seed)
    return rng.beta(alpha, beta, size=num_nodes)


# -------------------------------------------------------
# 3. Simulate social interactions (binary feedback)
# -------------------------------------------------------

def simulate_interactions(G, theta_true, interactions_per_edge=5, seed=42):
    """
    Simulates positive (1) or negative (0) feedback between neighbors.
    """
    rng = np.random.default_rng(seed)
    ratings = {}

    for u, v in G.edges():
        for _ in range(interactions_per_edge):
            y_vu = rng.binomial(1, theta_true[u])
            y_uv = rng.binomial(1, theta_true[v])

            ratings.setdefault((v, u), []).append(y_vu)
            ratings.setdefault((u, v), []).append(y_uv)

    return ratings


# -------------------------------------------------------
# 4. Bayesian Reputation Computation
# -------------------------------------------------------

def weight_function(reputation):
    """
    Weight assigned to a rating based on rater's reputation.
    """
    return reputation


def bayesian_reputation(
    G,
    ratings,
    alpha0=1.0,
    beta0=1.0,
    max_iters=50,
    tolerance=1e-4
):
    """
    Iterative Bayesian reputation computation.
    """
    num_nodes = G.number_of_nodes()
    R = np.full(num_nodes, alpha0 / (alpha0 + beta0))

    for iteration in range(max_iters):
        S = np.zeros(num_nodes)
        F = np.zeros(num_nodes)

        for (rater, rated), obs_list in ratings.items():
            if not G.has_edge(rater, rated):
                continue

            w = weight_function(R[rater])
            if w == 0:
                continue

            obs = np.array(obs_list)
            S[rated] += w * obs.sum()
            F[rated] += w * (len(obs) - obs.sum())

        alpha_post = alpha0 + S
        beta_post = beta0 + F
        R_new = alpha_post / (alpha_post + beta_post)

        delta = np.max(np.abs(R_new - R))
        R = R_new

        print(f"Iteration {iteration+1:02d} | max change = {delta:.6f}", flush=True)
        if delta < tolerance:
            break

    return R


# -------------------------------------------------------
# 5. Main execution
# -------------------------------------------------------

def main():
    NUM_NODES = 80
    EDGE_PROB = 0.06
    INTERACTIONS = 6

    print("Generating social network...", flush=True)
    G = generate_social_network(NUM_NODES, EDGE_PROB)

    print("Generating true reputations...", flush=True)
    theta_true = generate_true_reputation(NUM_NODES)

    print("Simulating social interactions...", flush=True)
    ratings = simulate_interactions(G, theta_true, INTERACTIONS)

    print("Computing Bayesian reputations...\n", flush=True)
    R_estimated = bayesian_reputation(G, ratings)

    print("\n--- Sample Results (First 10 Nodes) ---", flush=True)
    for i in range(10):
        print(
            f"User {i:02d} | "
            f"Estimated Reputation = {R_estimated[i]:.3f} | "
            f"True Reputation = {theta_true[i]:.3f}",
            flush=True
        )

    corr = np.corrcoef(theta_true, R_estimated)[0, 1]
    print(f"\nCorrelation between true and estimated reputation: {corr:.3f}", flush=True)


if __name__ == "__main__":
    main()
