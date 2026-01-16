"""
Bayesian Belief Network based Node Reputation System
---------------------------------------------------
- Builds a complex network
- Extracts node features from topology
- Uses Bayesian Belief Network (pgmpy)
- Iteratively propagates reputation
- Visualizes final reputation
"""

# =========================
# 1. Imports
# =========================
import random
import networkx as nx
import matplotlib.pyplot as plt

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# =========================
# 2. Create Network
# =========================
NUM_NODES = 6
EDGE_PROB = 0.6
ITERATIONS = 5

G = nx.erdos_renyi_graph(n=NUM_NODES, p=EDGE_PROB, seed=42)

# Add edge weights
for u, v in G.edges():
    G[u][v]['weight'] = round(random.uniform(0.1, 1.0), 2)

# =========================
# 3. Derive Node Features
# =========================
for node in G.nodes():
    # Activity from degree
    deg = G.degree(node)
    if deg <= 1:
        activity = 'low'
    elif deg == 2:
        activity = 'medium'
    else:
        activity = 'high'
    G.nodes[node]['Activity'] = activity

    # Connection strength from average edge weight
    weights = [G[node][nbr]['weight'] for nbr in G.neighbors(node)]
    avg_weight = sum(weights) / len(weights) if weights else 0
    G.nodes[node]['Connection'] = 'strong' if avg_weight > 0.5 else 'weak'

    # Initialize feedback
    G.nodes[node]['Feedback'] = 'neutral'

# =========================
# 4. Bayesian Belief Network
# =========================
model = DiscreteBayesianNetwork([
    ('Activity', 'Reputation'),
    ('Feedback', 'Reputation'),
    ('Connection', 'Reputation')
])

cpd_activity = TabularCPD(
    'Activity', 3, [[0.3], [0.5], [0.2]],
    state_names={'Activity': ['low', 'medium', 'high']}
)

cpd_feedback = TabularCPD(
    'Feedback', 3, [[0.2], [0.5], [0.3]],
    state_names={'Feedback': ['negative', 'neutral', 'positive']}
)

cpd_connection = TabularCPD(
    'Connection', 2, [[0.4], [0.6]],
    state_names={'Connection': ['weak', 'strong']}
)

cpd_reputation = TabularCPD(
    'Reputation', 3,
    values=[
        # low reputation
        [0.9, 0.7, 0.6, 0.4, 0.3, 0.2,
         0.1, 0.05, 0.01, 0.05, 0.02, 0.01,
         0.05, 0.02, 0.01, 0.01, 0.02, 0.01],
        # medium reputation
        [0.08, 0.2, 0.3, 0.4, 0.4, 0.3,
         0.3, 0.2, 0.2, 0.2, 0.1, 0.05,
         0.1, 0.05, 0.03, 0.02, 0.03, 0.02],
        # high reputation
        [0.02, 0.1, 0.1, 0.2, 0.3, 0.5,
         0.6, 0.75, 0.79, 0.75, 0.88, 0.94,
         0.85, 0.93, 0.96, 0.97, 0.95, 0.97]
    ],
    evidence=['Activity', 'Feedback', 'Connection'],
    evidence_card=[3, 3, 2],
    state_names={
        'Reputation': ['low', 'medium', 'high'],
        'Activity': ['low', 'medium', 'high'],
        'Feedback': ['negative', 'neutral', 'positive'],
        'Connection': ['weak', 'strong']
    }
)

model.add_cpds(cpd_activity, cpd_feedback, cpd_connection, cpd_reputation)
assert model.check_model()

infer = VariableElimination(model)

# =========================
# 5. Iterative Reputation Propagation
# =========================
rep_scores = {'low': 1, 'medium': 2, 'high': 3}

node_reputation = {node: 2 for node in G.nodes()}  # start medium

for _ in range(ITERATIONS):
    new_rep = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:
            avg_rep = sum(node_reputation[n] for n in neighbors) / len(neighbors)
            if avg_rep < 1.5:
                feedback = 'negative'
            elif avg_rep < 2.5:
                feedback = 'neutral'
            else:
                feedback = 'positive'
        else:
            feedback = 'neutral'

        G.nodes[node]['Feedback'] = feedback

        q = infer.query(
            variables=['Reputation'],
            evidence={
                'Activity': G.nodes[node]['Activity'],
                'Feedback': feedback,
                'Connection': G.nodes[node]['Connection']
            }
        )

        expected_rep = sum(
            q.values[i] * rep_scores[q.state_names['Reputation'][i]]
            for i in range(len(q.values))
        )
        new_rep[node] = expected_rep

    node_reputation = new_rep

# =========================
# 6. Print Final Scores
# =========================
print("\nFinal Node Reputation Scores:\n", flush=True)
for node, rep in node_reputation.items():
    print(f"Node {node}: {rep:.2f}", flush=True)

# =========================
# 7. Visualization
# =========================
def rep_color(r):
    if r < 1.5:
        return 'red'
    elif r < 2.5:
        return 'orange'
    else:
        return 'green'

node_colors = [rep_color(node_reputation[n]) for n in G.nodes()]
node_sizes = [300 + 700 * (node_reputation[n] - 1) / 2 for n in G.nodes()]

pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(8, 6))
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
nx.draw_networkx_edges(G, pos, alpha=0.6)
nx.draw_networkx_labels(G, pos)
plt.title("Bayesian Node Reputation in Complex Network")
plt.axis("off")
plt.show()
