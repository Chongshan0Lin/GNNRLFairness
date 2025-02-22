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


        # print("feature_matrix.shape: ",self.feature_matrix.shape)
        # print("nnodes: ",self.nnodes)
        # print("nclasses: ",self.nclasses)
        # print("hfeatures: ",self.hfeatures)

        self.model = GCN(in_features=self.nfeatures, hidden_features = self.hfeatures, out_features=self.nclasses, dropout=0.5)
        self.model.to(device)

        # if torch.cuda.is_available():
        #     print("CUDA is available. PyTorch can use the GPU.")
        #     print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        #     print(f"Current CUDA device: {torch.cuda.current_device()}")
        #     print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        # else:
        #     print("CUDA is not available. PyTorch will use the CPU.")
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # device = torch.device(f"cuda:{gpu_index}"if torch.cuda.is_available() else "cpu")
        # self.model.to(device)

        # params = list(self.model.parameters())
        # param_count = len(params)
        # print(f"Total parameters in model: {param_count}")
        # for idx, param in enumerate(params):
        #     print(f"Parameter {idx}: Shape {param.shape}, requires_grad={param.requires_grad}")

        # print("number of parameter to optimizer",self.model.parameters().__sizeof__())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        # print("Optimizer initialized with model parameters.")

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

        DP = demographic_parity(predictions=pred_test, sens=self.sens[self.idx_test])
        EOd = equality_of_odds(predictions=pred_test, labels=self.labels[self.idx_test], sens=self.sens[self.idx_test])
        CDP = conditional_demographic_parity(predictions=pred_test, labels=self.labels[self.idx_test], sens=self.sens[self.idx_test])
        surrogate = fair_metric(pred_test, self.labels[self.idx_test], self.sens[self.idx_test])
        print(f"Demographic Parity: {DP:.4f}, Equality of Odds: {EOd:.4f}, Conditional DP: {CDP:.4f}")
        return surrogate[0], DP.item(), EOd.item(), CDP.item()

    # def fair_metric(pred, labels, sens):
    #     # pred = pred.detach().cpu().numpy()
    #     # labels = labels.detach().cpu().numpy()
    #     # sens = sens.detach().cpu().numpy()
        
    #     # idx_s0 = sens == 0
    #     # idx_s1 = sens == 1
    #     # idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    #     # idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
        
    #     # epsilon = 1e-8
    #     # parity = abs(pred[idx_s0].sum() / (idx_s0.sum() + epsilon) - pred[idx_s1].sum() / (idx_s1.sum() + epsilon))
    #     # equality = abs(pred[idx_s0_y1].sum() / (idx_s0_y1.sum() + epsilon) - pred[idx_s1_y1].sum() / (idx_s1_y1.sum() + epsilon))
        
    #     # return parity, equality
    # # Move tensors to CPU and convert to NumPy arrays
    #     pred_np = pred.detach().cpu().numpy()
    #     labels_np = labels.detach().cpu().numpy()
    #     sens_np = sens.detach().cpu().numpy()
        
    #     # Compute boolean indices
    #     idx_s0 = sens_np == 0
    #     idx_s1 = sens_np == 1
    #     idx_s0_y1 = np.bitwise_and(idx_s0, labels_np == 1)
    #     idx_s1_y1 = np.bitwise_and(idx_s1, labels_np == 1)
        
    #     # Calculate sums with epsilon to avoid division by zero
    #     epsilon = 1e-8
    #     sum_s0 = np.sum(idx_s0) + epsilon
    #     sum_s1 = np.sum(idx_s1) + epsilon
    #     sum_s0_y1 = np.sum(idx_s0_y1) + epsilon
    #     sum_s1_y1 = np.sum(idx_s1_y1) + epsilon
        
    #     # Calculate parity and equality using NumPy operations
    #     parity = abs(np.sum(pred_np[idx_s0]) / sum_s0 - np.sum(pred_np[idx_s1]) / sum_s1)
    #     equality = abs(np.sum(pred_np[idx_s0_y1]) / sum_s0_y1 - np.sum(pred_np[idx_s1_y1]) / sum_s1_y1)
        
    #     return parity, equality



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

        # Update the adj_norm correspondingly
        self.adj_norm = normalize_adjacency(self.adj_matrix).detach().numpy()
    
    def update_adj_matrix(self, adj_matrix):
        # self.adj_matrix = adj_matrix
        # self.adj_norm = normalize_adjacency(self.adj_matrix).detach().numpy()

        device = torch.device(f"cuda:{gpu_index}"if torch.cuda.is_available() else "cpu")
        self.adj_matrix = adj_matrix.to(device) 
        self.adj_norm = normalize_adjacency(self.adj_matrix).detach()


s = victim()
s.train()
s.evaluate()