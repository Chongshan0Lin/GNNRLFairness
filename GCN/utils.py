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
    print(adj)
    print(type(adj))
    adj = torch.Tensor.to_dense(adj)
    adj = np.add(adj, torch.eye(len(adj[0])))
    # Compute degree matrix
    rowsum = adj.sum(1)
    d_inv_sqrt = rowsum.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0

    # Construct D^{-1/2} * A * D^{-1/2}
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

def graph_to_adj(graph):

    A = nx.adjacency_matrix(graph)
    adj_matrix = A.toarray()

    return adj_matrix