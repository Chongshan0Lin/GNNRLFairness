from .data_loading import graph_loading, feature_loading, label_loading, loading_facebook_dataset
import networkx as nx
import pandas as pd
import numpy as np
import torch
from .model import GCN
from .utils import normalize_adjacency
from .utils import demographic_parity, conditional_demographic_parity, equality_of_odds
from .utils import fair_metric
from .kernel_estimator import KernelEstimator
import torch.nn.functional as F
EPOCH = 100
gpu_index = 2
from torch.nn import functional as F


device = torch.device(f"cuda:{gpu_index}"if torch.cuda.is_available() else "cpu")

class victim:

    def __init__(self):
        self.adj_matrix, self.feature_matrix, self.labels, self.idx_train, self.idx_val, self.idx_test, self.sens = loading_facebook_dataset()
        self.nnodes = self.feature_matrix.shape[0]
        self.nfeatures = self.feature_matrix.shape[1]
        self.nclasses = int(self.labels.max() + 1)
        self.hfeatures = int((self.nfeatures * 2) // 3 + self.nclasses)


        self.labels = self.labels.to(device)
        self.sens = self.sens.to(device)
        if isinstance(self.idx_train, torch.Tensor):
            self.idx_train = self.idx_train.to(device)
        if isinstance(self.idx_val, torch.Tensor):
            self.idx_val = self.idx_val.to(device)
        if isinstance(self.idx_test, torch.Tensor):
            self.idx_test = self.idx_test.to(device)



        self.model = GCN(in_features=self.nfeatures, hidden_features = self.hfeatures, out_features=self.nclasses, dropout=0.5)
        self.model.to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)

        self.adj_norm = normalize_adjacency(self.adj_matrix).detach().numpy()



    def train(self):

        self.model.train()
        for epoch in range(EPOCH):

            self.optimizer.zero_grad()
            # print(feature_matrix.shape)
            # print(feature_matrix.shape[0])
            # print(feature_matrix.shape[1])
            output = self.model(self.feature_matrix, self.adj_norm)

            # Compute loss only over training nodes
            loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
            loss_train.backward()
            self.optimizer.step()

            # Optional: monitor validation accuracy
            with torch.no_grad():
                self.model.eval()
                output_val = self.model(self.feature_matrix, self.adj_norm)
                loss_val = F.nll_loss(output_val[self.idx_val], self.labels[self.idx_val])
                pred_val = output_val[self.idx_val].max(1)[1]
                acc_val  = pred_val.eq(self.labels[self.idx_val]).sum().item() / self.idx_val.size(0)
                self.model.train()
            # print("Episode")
            if (epoch+1) % 100 == 0:
                print(
                    f"Epoch: {epoch:03d}, "
                    f"Train Loss: {loss_train.item():.4f}, "
                    f"Val Loss: {loss_val.item():.4f}, "
                    f"Val Acc: {acc_val:.4f}"
                )
        # return 0
    
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            output_test = self.model(self.feature_matrix, self.adj_norm)
            loss_test = F.nll_loss(output_test[self.idx_test], self.labels[self.idx_test])
            pred_test = output_test[self.idx_test].max(1)[1]
            acc_test  = pred_test.eq(self.labels[self.idx_test]).sum().item() / self.idx_test.size(0)

        print(f"Test Loss: {loss_test.item():.4f}, Test Accuracy: {acc_test:.4f}")

        """
        Fairness Evauation
        """
        
        losses  = fair_metric(pred_test, self.labels[self.idx_test], self.sens[self.idx_test])
        DP, EOd = losses[0], losses[1]
        print(f"Demographic Parity: {DP:.4f}, Equality of Odds: {EOd:.4f}")
        return DP, EOd



    def change_edge(self, node1, node2):
        """
        Change the connection between node1 and node2
        If there already exists an connection between these two, remove it.
        Otherwise, connect them.
        """
        self.adj_matrix = self.adj_matrix.to_dense()
        if self.adj_matrix[node1][node2]:
            self.adj_matrix[node1][node2] = 0
        else:
            self.adj_matrix[node1][node2] = 1

        self.adj_norm = normalize_adjacency(self.adj_matrix).detach().numpy()
    
    def update_adj_matrix(self, adj_matrix):

        device = torch.device(f"cuda:{gpu_index}"if torch.cuda.is_available() else "cpu")
        self.adj_matrix = adj_matrix.to(device) 
        self.adj_norm = normalize_adjacency(self.adj_matrix).detach()


s = victim()
s.train()
s.evaluate()