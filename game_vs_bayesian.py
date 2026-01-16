"""
Bayesian vs Game-Theoretic Reputation Models in Undirected Social Networks
=======================================================================

This file implements and compares:
1. Bayesian Reputation Model (probabilistic belief updating)
2. Game-Theoretic Reputation Model (strategic agents with utilities)

The same social network and interactions are used for fair comparison.

Author: (Add your name)
"""

import numpy as np
import networkx as nx

# --------------------------------------------------
# 1. Social Network
# --------------------------------------------------

def create_network(n=30, p=0.1, seed=42):
    """Create a small connected network"""
    np.random.seed(seed)
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    
    # Ensure connected by adding edges between components
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = list(components[i])[0]
            v = list(components[i+1])[0]
            G.add_edge(u, v)
    
    return G

# --------------------------------------------------
# 2. Ground Truth Reputation
# --------------------------------------------------

def generate_true_reputation(n, seed=42):
    rng = np.random.default_rng(seed)
    return rng.beta(4, 2, size=n)

# --------------------------------------------------
# 3. Interaction Simulation (shared)
# --------------------------------------------------

def simulate_interactions(G, theta, k=6, seed=42):
    rng = np.random.default_rng(seed)
    feedback = {}
    for u, v in G.edges():
        for _ in range(k):
            feedback.setdefault((u, v), []).append(rng.binomial(1, theta[v]))
            feedback.setdefault((v, u), []).append(rng.binomial(1, theta[u]))
    return feedback

# --------------------------------------------------
# 4. Bayesian Reputation Model
# --------------------------------------------------

def bayesian_model(G, feedback, alpha0=1, beta0=1, tol=1e-4, max_iter=50):
    n = G.number_of_nodes()
    R = np.full(n, alpha0 / (alpha0 + beta0))

    for it in range(max_iter):
        S, F = np.zeros(n), np.zeros(n)
        for (i, j), obs in feedback.items():
            w = R[i]
            obs = np.array(obs)
            S[j] += w * obs.sum()
            F[j] += w * (len(obs) - obs.sum())
        R_new = (alpha0 + S) / (alpha0 + beta0 + S + F)
        if np.max(np.abs(R_new - R)) < tol:
            break
        R = R_new
    return R

# --------------------------------------------------
# 5. Game-Theoretic Reputation Model
# --------------------------------------------------

def game_theoretic_model(G, theta, benefit=1.0, lie_cost=0.3, max_iter=30):
    """
    Agents choose honest or dishonest behavior to maximize utility.
    Utility: U_i = benefit * reputation - cost_of_lying
    """
    n = G.number_of_nodes()
    reputation = np.full(n, 0.5)

    for _ in range(max_iter):
        new_rep = np.zeros(n)
        for i in range(n):
            neighbors = list(G.neighbors(i))
            if not neighbors:
                new_rep[i] = reputation[i]
                continue

            honest_payoff = benefit * reputation[i]
            dishonest_payoff = benefit * (1 - reputation[i]) - lie_cost

            action = 1 if honest_payoff >= dishonest_payoff else 0

            signals = []
            for j in neighbors:
                signal = np.random.binomial(1, theta[i]) if action == 1 else np.random.binomial(1, 1 - theta[i])
                signals.append(signal)

            new_rep[i] = np.mean(signals)

        reputation = 0.5 * reputation + 0.5 * new_rep

    return reputation

# --------------------------------------------------
# 6. Evaluation
# --------------------------------------------------

def evaluate(true_r, est_r):
    return np.corrcoef(true_r, est_r)[0, 1]

# --------------------------------------------------
# 7. Main Comparison
# --------------------------------------------------

def main():
    G = create_network()
    theta_true = generate_true_reputation(G.number_of_nodes())
    feedback = simulate_interactions(G, theta_true)

    print("Running Bayesian model...", flush=True)
    R_bayes = bayesian_model(G, feedback)

    print("Running Game-Theoretic model...", flush=True)
    R_game = game_theoretic_model(G, theta_true)

    corr_bayes = evaluate(theta_true, R_bayes)
    corr_game = evaluate(theta_true, R_game)

    print("\n--- Performance Comparison ---", flush=True)
    print(f"Bayesian Model Correlation: {corr_bayes:.3f}", flush=True)
    print(f"Game-Theoretic Model Correlation: {corr_game:.3f}", flush=True)

    print("\nSample Results (First 10 Nodes)", flush=True)
    for i in range(10):
        print(f"User {i:02d} | True={theta_true[i]:.3f} | Bayesian={R_bayes[i]:.3f} | Game={R_game[i]:.3f}", flush=True)


if __name__ == "__main__":
    main()
