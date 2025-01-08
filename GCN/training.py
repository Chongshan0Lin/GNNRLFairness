from data_loading import graph_loading, feature_loading, label_loading, loading_facebook_dataset
import networkx as nx
import pandas as pd
import numpy as np
import torch
from utils import graph_to_adj
from model import GCN
from utils import normalize_adjacency
import torch.nn.functional as F



# graph = graph_loading()
# num_nodes = graph.number_of_nodes()
# num_edges = graph.size()
# adj_matrix = graph_to_adj(graph)
# labels, label_mapping = label_loading()
# feature_matrix = feature_loading()
# np.random.seed(2022)

# labels = torch.from_numpy(labels)

# in_features = len(feature_matrix[0])
# out_features = len(label_mapping)
# hidden_features = (in_features * 2) // 3 + out_features


# idx_train = torch.arange(0, (num_nodes * 6) // 10)  # first 6 nodes are train
# idx_val   = torch.arange((num_nodes * 6) // 10, (num_nodes * 8) // 10)  # next 2 nodes for validation
# idx_test  = torch.arange((num_nodes * 8) // 10, num_nodes) # last 2 nodes for test

adj_matrix, feature_matrix, labels, idx_train, idx_val, idx_test, sens = loading_facebook_dataset()
nnodes = feature_matrix.shape[0]
nfeatures = feature_matrix.shape[1]
nclasses = int(labels.max() + 1)
hfeatures = int((nfeatures * 2) // 3 + nclasses)
print("feature_matrix.shape: ",feature_matrix.shape)
print("nnodes: ",nnodes)
print("nclasses: ",nclasses)
print("hfeatures: ",hfeatures)


model = GCN(in_features=nfeatures, hidden_features = hfeatures, out_features=nclasses, dropout=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

adj_norm = normalize_adjacency(adj_matrix).detach().numpy()

model.train()
for epoch in range(100):

    optimizer.zero_grad()
    # print(feature_matrix.shape)
    # print(feature_matrix.shape[0])
    # print(feature_matrix.shape[1])
    output = model(feature_matrix, adj_norm)

    # Compute loss only over training nodes
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # Optional: monitor validation accuracy
    with torch.no_grad():
        model.eval()
        output_val = model(feature_matrix, adj_norm)
        loss_val = F.nll_loss(output_val[idx_val], labels[idx_val])
        pred_val = output_val[idx_val].max(1)[1]
        acc_val  = pred_val.eq(labels[idx_val]).sum().item() / idx_val.size(0)
        model.train()

    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch:03d}, "
            f"Train Loss: {loss_train.item():.4f}, "
            f"Val Loss: {loss_val.item():.4f}, "
            f"Val Acc: {acc_val:.4f}"
        )

# -------------------------
# 5.5 Testing
# -------------------------
model.eval()
with torch.no_grad():
    output_test = model(feature_matrix, adj_norm)
    loss_test = F.nll_loss(output_test[idx_test], labels[idx_test])
    pred_test = output_test[idx_test].max(1)[1]
    acc_test  = pred_test.eq(labels[idx_test]).sum().item() / idx_test.size(0)

print(f"Test Loss: {loss_test.item():.4f}, Test Accuracy: {acc_test:.4f}")
