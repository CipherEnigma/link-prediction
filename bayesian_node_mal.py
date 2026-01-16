"""
Bayesian Node Reputation in Sparse Complex Network
--------------------------------------------------
- Reduced connectivity
- Injected malicious nodes
- Bayesian Belief Network (pgmpy)
- Iterative reputation propagation
- Visualization of trust collapse
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
# 2. Create Sparse Network
# =========================
NUM_NODES = 10
EDGE_PROB = 0.25       # 👈 reduced connectivity
ITERATIONS = 6

G = nx.erdos_renyi_graph(n=NUM_NODES, p=EDGE_PROB, seed=42)

# Add random edge weights
for u, v in G.edges():
    G[u][v]['weight'] = round(random.uniform(0.1, 1.0), 2)

# =========================
# 3. Inject Malicious Nodes
# =========================
MALICIOUS_NODES = random.sample(list(G.nodes()), k=3)
print("Malicious nodes:", MALICIOUS_NODES, flush=True)

# =========================
# 4. Derive Node Features
# =========================
for node in G.nodes():
    deg = G.degree(node)

    # Activity
    if node in MALICIOUS_NODES:
        activity = 'low'
    else:
        if deg <= 1:
            activity = 'low'
        elif deg == 2:
            activity = 'medium'
        else:
            activity = 'high'
    G.nodes[node]['Activity'] = activity

    # Connection strength
    weights = [G[node][nbr]['weight'] for nbr in G.neighbors(node)]
    avg_weight = sum(weights)/len(weights) if weights else 0

    if node in MALICIOUS_NODES:
        G.nodes[node]['Connection'] = 'weak'
    else:
        G.nodes[node]['Connection'] = 'strong' if avg_weight > 0.5 else 'weak'

    # Initial feedback
    G.nodes[node]['Feedback'] = 'negative' if node in MALICIOUS_NODES else 'neutral'

# =========================
# 5. Bayesian Belief Network
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
        # Low reputation
        [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.15,0.1,0.05,
         0.2,0.15,0.1,0.1,0.08,0.05],
        # Medium reputation
        [0.08,0.15,0.2,0.25,0.3,0.35,0.4,0.4,0.45,0.4,0.35,0.25,
         0.35,0.35,0.3,0.25,0.22,0.2],
        # High reputation
        [0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.45,0.45,0.55,0.7,
         0.45,0.5,0.6,0.65,0.7,0.75]
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
# 6. Iterative Reputation Propagation
# =========================
rep_scores = {'low': 1, 'medium': 2, 'high': 3}

# Initialize reputations as medium
node_reputation = {node: 2 for node in G.nodes()}

for _ in range(ITERATIONS):
    new_rep = {}

    for node in G.nodes():
        neighbors = list(G.neighbors(node))

        # Feedback logic
        if node in MALICIOUS_NODES:
            feedback = 'negative'
        else:
            if neighbors:
                avg_rep = sum(node_reputation[n] for n in neighbors) / len(neighbors)
                malicious_neighbors = sum(1 for n in neighbors if n in MALICIOUS_NODES)

                if malicious_neighbors >= len(neighbors)/2:
                    feedback = 'negative'
                elif avg_rep < 1.5:
                    feedback = 'negative'
                elif avg_rep < 2.5:
                    feedback = 'neutral'
                else:
                    feedback = 'positive'
            else:
                feedback = 'neutral'

        G.nodes[node]['Feedback'] = feedback

        # Bayesian inference
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
# 7. Print Final Reputation
# =========================
print("\nFinal Node Reputation Scores:\n", flush=True)
for node, rep in node_reputation.items():
    label = " (MALICIOUS)" if node in MALICIOUS_NODES else ""
    print(f"Node {node}: {rep:.2f}{label}", flush=True)

# =========================
# 8. Visualization
# =========================
def node_color(node, rep):
    if node in MALICIOUS_NODES:
        return 'black'
    if rep < 1.5:
        return 'red'
    elif rep < 2.5:
        return 'orange'
    else:
        return 'green'

node_colors = [node_color(n, node_reputation[n]) for n in G.nodes()]
node_sizes = [300 + 700 * (node_reputation[n]-1)/2 for n in G.nodes()]

pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(9, 7))
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
nx.draw_networkx_edges(G, pos, alpha=0.6)
nx.draw_networkx_labels(G, pos, font_color='white')
plt.title("Bayesian Node Reputation with Malicious Nodes (Sparse Network)")
plt.axis("off")
plt.show()
