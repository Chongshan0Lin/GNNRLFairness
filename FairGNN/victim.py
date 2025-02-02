from .data_loading import graph_loading, feature_loading, label_loading, loading_facebook_dataset
import networkx as nx
import pandas as pd
import numpy as np
import torch
# from .model import GCN
from .FairGNN import FairGNN
from .utils import normalize_adjacency
from .utils import demographic_parity, conditional_demographic_parity, equality_of_odds
from .utils import fair_metric
import torch.nn.functional as F
import dgl
import time
from sklearn.metrics import roc_auc_score
import scipy.sparse as sp

EPOCH = 100
gpu_index = 2
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
        # def __init__(self, nfeat, nhid, nclass, weight_decay, lr, dropout, alpha, beta):


        self.model = FairGNN(nfeat=self.nfeatures, nhid=self.hfeatures, nclass=self.nclasses, dropout=0.5, lr = 1e-3, weight_decay=1e-5, alpha=60, beta=10)
        self.model.to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)

        self.adj_norm = normalize_adjacency(self.adj_matrix).detach().numpy()

    # def train(self):
    def train(self):
        """
        New training loop based on your first code chunk.
        (Some minor adaptations were made so that the victim’s data attributes are used.)
        """
        # --- Setup ---
        # Create a DGL graph from the (assumed scipy sparse) adjacency matrix.
        # G = dgl.DGLGraph()
        # G = dgl.graph()
        # adj_np = self.adj_matrix.to_dense().cpu().numpy()  # Ensure the tensor is on CPU and convert to NumPy.
        # adj_sp = sp.csr_matrix(adj_np)          # Create a SciPy CSR sparse matrix.
        # G = dgl.from_scipy(adj_sp)              # Now create the DGL graph.
        # G = dgl.add_self_loop(G)
        # g = dgl.from_scipy(adj_norm)
        self.G = dgl.from_scipy(self.adj_norm)
        self.G = self.G.to(device)

        # Get the data from self.
        features = self.feature_matrix.to(device)
        labels = self.labels
        idx_train = self.idx_train
        idx_val = self.idx_val
        idx_test = self.idx_test
        sens = self.sens
        # If a separate sensitive attribute training index is needed, here we re-use idx_train.
        idx_sens_train = self.idx_train

        # Define training hyperparameters.
        epochs = 2000
        acc_threshold = 0.688
        roc_threshold = 0.745

        best_result = {}
        best_fair = float('inf')
        t_total = time.time()

        # --- Define a local fairness-metric function ---
        def fair_metric_new(output, idx, labels, sens):
            """
            Computes parity and equality differences.
            (This function assumes binary sensitive attribute and binary classification.)
            """
            # Extract ground-truth for the selected indices.
            val_y = labels[idx].cpu().numpy()
            # Get sensitive attribute values for the given indices.
            idx_cpu = idx.cpu().numpy()
            sens_cpu = sens.cpu().numpy()
            idx_s0 = sens_cpu[idx_cpu] == 0
            idx_s1 = sens_cpu[idx_cpu] == 1

            # For equality of opportunity, focus on the positive label.
            idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
            idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)

            # Here we assume that the raw output is such that positive predictions exceed 0.
            pred_y = (output[idx].squeeze() > 0).type_as(labels).cpu().numpy()
            # Use a small epsilon to avoid division-by-zero.
            eps = 1e-8
            parity = abs((sum(pred_y[idx_s0]) / (sum(idx_s0) + eps)) - (sum(pred_y[idx_s1]) / (sum(idx_s1) + eps)))
            equality = abs((sum(pred_y[idx_s0_y1]) / (sum(idx_s0_y1) + eps)) - (sum(pred_y[idx_s1_y1]) / (sum(idx_s1_y1) + eps)))
            return parity, equality

        # --- Training loop ---
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            # Perform one optimization step.
            # (Your FairGNN model should implement an optimize() method that uses G, features, etc.)
            self.model.optimize(self.G, features, labels, idx_train, sens, idx_sens_train)

            # Retrieve loss components (assumed to be stored as attributes by optimize()).
            cov = self.model.cov
            cls_loss = self.model.cls_loss
            adv_loss = self.model.adv_loss

            # Evaluate the model.
            self.model.eval()
            output, s = self.model(self.G, features)

            # Compute validation accuracy.
            pred_val = output[idx_val].max(1)[1]
            acc_val = (pred_val == labels[idx_val]).float().mean()
            try:
                roc_val = roc_auc_score(labels[idx_val].cpu().numpy(),
                                        output[idx_val].detach().cpu().numpy())
            except Exception as e:
                roc_val = 0.0

            # Compute fairness metrics on the validation set.
            parity_val, equality_val = fair_metric_new(output, idx_val, labels, sens)

            # Compute test set metrics.
            pred_test = output[idx_test].max(1)[1]
            acc_test = (pred_test == labels[idx_test]).float().mean()
            try:
                roc_test = roc_auc_score(labels[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy())
            except Exception as e:
                roc_test = 0.0
            parity, equality = fair_metric_new(output, idx_test, labels, sens)

            # Compute the sensitive attribute prediction accuracy (from the adversary output `s`).
            pred_sens = s[idx_test].max(1)[1]
            acc_sens = (pred_sens == sens[idx_test]).float().mean()

            # Check if the validation metrics meet the thresholds.
            if acc_val.item() > acc_threshold and roc_val > roc_threshold:
                if best_fair > (parity_val + equality_val):
                    best_fair = parity_val + equality_val
                    best_result['acc'] = acc_test.item()
                    best_result['roc'] = roc_test
                    best_result['parity'] = parity
                    best_result['equality'] = equality

                print("=================================")
                print('Epoch: {:04d}'.format(epoch + 1),
                      'cov: {:.4f}'.format(cov.item()),
                      'cls: {:.4f}'.format(cls_loss.item()),
                      'adv: {:.4f}'.format(adv_loss.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      "roc_val: {:.4f}".format(roc_val),
                      "parity_val: {:.4f}".format(parity_val),
                      "equality: {:.4f}".format(equality_val))
                print("Test:",
                      "accuracy: {:.4f}".format(acc_test.item()),
                      "roc: {:.4f}".format(roc_test),
                      "acc_sens: {:.4f}".format(acc_sens.item()),
                      "parity: {:.4f}".format(parity),
                      "equality: {:.4f}".format(equality))
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print('============performance on test set=============')
        if best_result:
            print("Test:",
                  "accuracy: {:.4f}".format(best_result['acc']),
                  "roc: {:.4f}".format(best_result['roc']),
                  "acc_sens: {:.4f}".format(acc_sens.item()),
                  "parity: {:.4f}".format(best_result['parity']),
                  "equality: {:.4f}".format(best_result['equality']))
        else:
            print("Please set smaller acc/roc thresholds")


    def evaluate(self):
        self.model.eval()
        with torch.no_grad():

            features = self.feature_matrix.to(device)
            # output_test = self.model(self.feature_matrix, self.adj_norm)
            output_test = self.model(self.G, features)
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
        return surrogate, DP.item(), EOd.item(), CDP.item()


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