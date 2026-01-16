"""
Link Prediction using Matrix Factorization
All results saved into ONE CSV file
"""

# =======================
# 1. Imports
# =======================
import torch
import torch.nn as nn
import torch.optim as optim
import random
import csv
from sklearn.metrics import roc_auc_score, average_precision_score

random.seed(42)
torch.manual_seed(42)

# =======================
# 2. Adjacency Matrix
# =======================
# Larger graph with 10 nodes
A = torch.tensor([
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
], dtype=torch.float32)

N = A.shape[0]

# =======================
# 3. Create Edge Lists
# =======================
positive_edges = [(i, j) for i in range(N) for j in range(i+1, N) if A[i, j] == 1]
negative_edges = [(i, j) for i in range(N) for j in range(i+1, N) if A[i, j] == 0]

random.shuffle(positive_edges)
split = int(0.8 * len(positive_edges))

train_pos = positive_edges[:split]
test_pos = positive_edges[split:]

random.shuffle(negative_edges)
train_neg = negative_edges[:len(train_pos)]
test_neg = negative_edges[len(train_pos):len(train_pos)+len(test_pos)]

# =======================
# 4. MF Model
# =======================
class MFLinkPredictor(nn.Module):
    def __init__(self, num_nodes, k):
        super().__init__()
        self.U = nn.Parameter(torch.randn(num_nodes, k))
        self.V = nn.Parameter(torch.randn(num_nodes, k))

    def forward(self):
        return torch.sigmoid(self.U @ self.V.T)

# =======================
# 5. Training
# =======================
model = MFLinkPredictor(N, k=4)
optimizer = optim.Adam(model.parameters(), lr=0.05)
criterion = nn.BCELoss()

epochs = 2000

for epoch in range(epochs):
    pred = model()

    pos_preds = torch.stack([pred[i, j] for (i, j) in train_pos])
    neg_preds = torch.stack([pred[i, j] for (i, j) in train_neg])

    preds = torch.cat([pos_preds, neg_preds])
    labels = torch.cat([
        torch.ones(len(pos_preds)),
        torch.zeros(len(neg_preds))
    ])

    loss = criterion(preds, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# =======================
# 6. Evaluation
# =======================
with torch.no_grad():
    pred = model()

    pos_test_preds = torch.stack([pred[i, j] for (i, j) in test_pos])
    neg_test_preds = torch.stack([pred[i, j] for (i, j) in test_neg])

    y_pred = torch.cat([pos_test_preds, neg_test_preds]).numpy()
    y_true = torch.cat([
        torch.ones(len(pos_test_preds)),
        torch.zeros(len(neg_test_preds))
    ]).numpy()

    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)

print(f"\nAUC: {auc:.4f}, AP: {ap:.4f}")

# =======================
# 7. Save EVERYTHING into ONE CSV
# =======================
output_file = "link_prediction_results.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "node_i",
        "node_j",
        "true_label",
        "predicted_probability",
        "AUC",
        "Average_Precision"
    ])

    # Positive test edges
    for (i, j), p in zip(test_pos, pos_test_preds):
        writer.writerow([i, j, 1, float(p), auc, ap])

    # Negative test edges
    for (i, j), p in zip(test_neg, neg_test_preds):
        writer.writerow([i, j, 0, float(p), auc, ap])

print(f"\nResults saved to: {output_file}")
