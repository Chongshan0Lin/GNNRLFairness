from GCN.victim import victim
from RLAttacker.RLattack import agent
import networkx as nx
import torch

target_model = victim()
# target_model.train()
# target_model.evaluate()

adj = target_model.adj_matrix
# print(adj)
G = nx.from_numpy_array(adj.to_dense().numpy())
# print(G)

attacker = agent(graph=G, feature_matrix=target_model.feature_matrix, labels=target_model.labels, state_dim=target_model.nnodes, idx_test=target_model.idx_test, idx_train=target_model.idx_train, epsilon=1, min_memory_step=200, budget=20)
attacker.train()
