from GCN.victim import victim
from RLAttacker.RLattack import agent
import networkx as nx
import torch

target_model = victim()

adj = target_model.adj_matrix
G = nx.from_numpy_array(adj.to_dense().numpy())

# Following the paper of Binchi et al., we set the budget to be 5% of the number of nodes.
budget = G.number_of_nodes()// 20

attacker = agent(graph=G, feature_matrix=target_model.feature_matrix, labels=target_model.labels, state_dim=target_model.nnodes, idx_test=target_model.idx_test, idx_train=target_model.idx_train, epsilon=0.1, min_memory_step=20, budget=budget)
attacker.train()
