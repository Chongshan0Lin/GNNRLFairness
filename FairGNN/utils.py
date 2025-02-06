import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx

def normalize_adjacency(adj):
    """
    Input: adj is a 2D torch.Tensor adjacency matrix
    Output: normalized adjacency matrix described in the paper Semi-Supervised Classification with Graph Convolutional Networks
    """

    # Add self-connections
    # print(adj)
    # print(type(adj))
    # adj = torch.Tensor.to_dense(adj)
    # adj = np.add(adj, torch.eye(len(adj[0])))
    # # Compute degree matrix
    # rowsum = adj.sum(1)
    # d_inv_sqrt = rowsum.pow(-0.5)
    # d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0

    # # Construct D^{-1/2} * A * D^{-1/2}
    # d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    # return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    if adj.is_sparse:
        adj = adj.to_dense()

    adj = adj + torch.eye(adj.size(0), device=adj.device)

    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0


    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

    adj_normalized = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    return adj_normalized


def graph_to_adj(graph):

    A = nx.adjacency_matrix(graph)
    adj_matrix = A.toarray()

    return adj_matrix

def demographic_parity(predictions, sens):

    minority = sens == 1
    majority = sens == 0
    positive = predictions == 1

    ma_pos = majority & positive
    mi_pos = minority & positive

    dp = abs(sum(ma_pos) / sum(majority) - sum(mi_pos) / sum(minority))
    return dp

def equality_of_odds(predictions, labels, sens):

    minority = sens == 1
    majority = sens == 0
    positive = predictions == 1
    ground_truth_true = labels == 1
    ground_truth_false = labels == 0

    ma_pos = majority & positive
    mi_pos = minority & positive

    ma_true = majority & ground_truth_true
    ma_false = majority & ground_truth_false
    mi_true = minority & ground_truth_true
    mi_false = minority & ground_truth_false

    ma_pos_true = ma_pos & ground_truth_true
    ma_pos_false = ma_pos & ground_truth_false
    mi_pos_true = mi_pos & ground_truth_true
    mi_pos_false = mi_pos & ground_truth_false
    print("Ma pos:", ma_pos)
    print(ma_pos_true)
    print("sum:",sum(ma_pos_true))
    # print(abs(sum(ma_pos_true) / sum(ma_true) - sum(mi_pos_true) / sum(mi_true)))
    EOd = abs(sum(ma_pos_true) / sum(ma_true) - sum(mi_pos_true) / sum(mi_true)) + abs(sum(ma_pos_false) / sum(ma_false) - sum(mi_pos_false) / sum(mi_false))

    return EOd

def conditional_demographic_parity(predictions, labels, sens):

    minority = sens == 1
    majority = sens == 0
    positive = predictions == 1
    negative = predictions == 0
    ground_truth_true = labels == 1

    ma_pos = majority & positive
    ma_neg = majority & negative
    mi_pos = minority & positive
    mi_neg = minority & negative

    ma_true = majority & ground_truth_true
    mi_true = minority & ground_truth_true

    ma_pos_true = ma_pos & ground_truth_true
    ma_neg_true = ma_neg & ground_truth_true
    mi_pos_true = mi_pos & ground_truth_true
    mi_neg_true = mi_neg & ground_truth_true
    # print("ma_pos_true:", sum(ma_pos_true))
    # print("mi_pos_true:", sum(mi_pos_true))
    # print("ma_neg_true:", sum(ma_neg_true))
    # print("mi_neg_true:", sum(mi_neg_true))
    
    CDP_pos = abs(sum(ma_pos_true) / sum(ma_true) - sum(mi_pos_true) / sum(mi_true))
    CDP_neg = abs(sum(ma_neg_true) / sum(ma_true) - sum(mi_neg_true) / sum(mi_true))
    # print("CDP_pos:", CDP_pos)
    # print("CDP_neg:", CDP_neg)

    return CDP_pos + CDP_neg

def fair_metric(pred, labels, sens):
    # pred = pred.detach().cpu().numpy()
    # labels = labels.detach().cpu().numpy()
    # sens = sens.detach().cpu().numpy()
    
    # idx_s0 = sens == 0
    # idx_s1 = sens == 1
    # idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    # idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    
    # epsilon = 1e-8
    # parity = abs(pred[idx_s0].sum() / (idx_s0.sum() + epsilon) - pred[idx_s1].sum() / (idx_s1.sum() + epsilon))
    # equality = abs(pred[idx_s0_y1].sum() / (idx_s0_y1.sum() + epsilon) - pred[idx_s1_y1].sum() / (idx_s1_y1.sum() + epsilon))
    
    # return parity, equality
# Move tensors to CPU and convert to NumPy arrays
    pred_np = pred.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    sens_np = sens.detach().cpu().numpy()
    
    # Compute boolean indices
    idx_s0 = sens_np == 0
    idx_s1 = sens_np == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels_np == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels_np == 1)
    
    # Calculate sums with epsilon to avoid division by zero
    epsilon = 1e-8
    sum_s0 = np.sum(idx_s0) + epsilon
    sum_s1 = np.sum(idx_s1) + epsilon
    sum_s0_y1 = np.sum(idx_s0_y1) + epsilon
    sum_s1_y1 = np.sum(idx_s1_y1) + epsilon
    
    # Calculate parity and equality using NumPy operations
    parity = abs(np.sum(pred_np[idx_s0]) / sum_s0 - np.sum(pred_np[idx_s1]) / sum_s1)
    equality = abs(np.sum(pred_np[idx_s0_y1]) / sum_s0_y1 - np.sum(pred_np[idx_s1_y1]) / sum_s1_y1)
    
    return parity, equality

