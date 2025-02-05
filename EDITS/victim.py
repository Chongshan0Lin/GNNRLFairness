from .data_loading import graph_loading, feature_loading, label_loading, loading_facebook_dataset
import networkx as nx
import pandas as pd
import numpy as np
import torch
from .gcn import GCN
import scipy.sparse as sp
from tqdm import tqdm
from .metrics import metric_wd
from .model import EDITS
# from torch_geometric.utils import convert
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
import os
import dgl
import shutil

from .utils import normalize_adjacency
from .utils import demographic_parity, conditional_demographic_parity, equality_of_odds
from .utils import fair_metric
import torch.nn.functional as F
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

        self.preprosessing()
        A_debiased, features = sp.load_npz('pre_processed/A_debiased.npz'), torch.load("pre_processed/X_debiased.pt", map_location=torch.device('cpu')).cpu().float()
        features = features[:, torch.nonzero(features.sum(axis=0)).squeeze()].detach()
        X_debiased = features.float().to(device)


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
        # def __init__(self, nfeat, nhid, nclass, dropout = 0.5):

        self.model = GCN(nfeat=X_debiased.shape[1], nhid = self.hfeatures, nclass=self.nclasses, dropout=0.5)

        # def __init__(self,  nfeat, node_num, nclass, nfeat_out, adj_lambda, layer_threshold=2, dropout=0.1, lr = 1e-3, weight_decay=1e-5):

        # self.model = EDITS(nfeat=self.nfeatures, node_num=self.nnodes, nclass=self.nclasses, nfeat_out=self.nclasses, adj_lambda=1e-1)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.adj_norm = normalize_adjacency(self.adj_matrix).detach().numpy()

    def train(self, pa = -1, eq = -1, test_f1 = -1, test_auc = -1, epoch = 100, val_loss = 1e5):
        best_val = val_loss
        adj_ori = self.adj_matrix
        if isinstance(adj_ori, torch.Tensor):
            adj_ori_scipy = self.tensor_to_scipy_sparse(adj_ori)
        else:
            adj_ori_scipy = adj_ori

        adj = self.normalize_scipy(self.adj_matrix)

        # Loading preprocessed data
        A_debiased, features = sp.load_npz('pre_processed/A_debiased.npz'), torch.load("pre_processed/X_debiased.pt", map_location=torch.device('cpu')).cpu().float()
        threshold_proportion = 0.015  # GCN: {credit: 0.02, german: 0.29, bail: 0.015}
        the_con1 = (A_debiased - adj_ori_scipy).A
        the_con1 = np.where(the_con1 > np.max(the_con1) * threshold_proportion, 1 + the_con1 * 0, the_con1)
        the_con1 = np.where(the_con1 < np.min(the_con1) * threshold_proportion, -1 + the_con1 * 0, the_con1)
        the_con1 = np.where(np.abs(the_con1) == 1, the_con1, the_con1 * 0)
        A_debiased = adj_ori_scipy + sp.coo_matrix(the_con1)
        assert A_debiased.max() == 1
        assert A_debiased.min() == 0
        features = features[:, torch.nonzero(features.sum(axis=0)).squeeze()].detach()
        A_debiased = self.normalize_scipy(A_debiased)
        
        sens = self.sens

        print("****************************After debiasing****************************")
        metric_wd(features, A_debiased, sens, 0.9, 0)
        metric_wd(features, A_debiased, sens, 0.9, 2)
        print("****************************************************************************")
        X_debiased = features.float().to(device)
        # edge_index = convert.from_scipy_sparse_matrix(A_debiased)[0].cuda()
        # Ensure your sparse matrix is in COO format.
        A_coo = A_debiased.tocoo()

        # Convert the row and column indices to torch tensors.
        row = torch.tensor(A_coo.row, dtype=torch.long)
        col = torch.tensor(A_coo.col, dtype=torch.long)

        # Stack them into an edge_index tensor of shape [2, num_edges].
        edge_index = torch.stack([row, col], dim=0).cuda()



        # model = GCN(nfeat=X_debiased.shape[1], nhid=self.hfeatures, nclass=self.labels.max().item()).float()
        # model = model.to(device)

        # Train model

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        idx_train = self.idx_train.to(device)
        labels = self.labels.to(device)
        idx_val =self.idx_val.to(device)

        num_nodes = A_coo.shape[0]
        g = dgl.graph((A_coo.row, A_coo.col), num_nodes=num_nodes)
        # g = g.to(device)  # Move the graph to the appropriate device.
        g = dgl.add_self_loop(g).to(device)

        # output = model(x=X_debiased, edge_index=torch.LongTensor(edge_index.cpu()).cuda())
        output = self.model(x=X_debiased, edge_index=g)
        preds = (output.squeeze() > 0).type_as(labels)
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train].long())
        
        loss_train.backward()
        self.optimizer.step()
        _, _ = fair_metric(preds[idx_train.cpu().numpy()].cpu().numpy(), labels[idx_train.cpu().numpy()].cpu().numpy(), sens[idx_train.cpu().numpy()].cpu().numpy())

        self.model.eval()
        # output = model(x=X_debiased, edge_index=torch.LongTensor(edge_index.cpu()).cuda())
        output = self.model(x=X_debiased, edge_index=g)
        preds = (output.squeeze() > 0).type_as(labels)
        loss_val = F.cross_entropy(output[idx_train], labels[idx_train].long())

        print("Val_loss:", loss_val)
        if loss_val <= best_val:
            best_val = loss_val


        # if loss_val < valsloss:
            # The problem might be here: it always takes the minimal loss_val
            val_loss = loss_val.data
            # print("New val_loss:", val_loss)
            pa, eq, test_f1, test_auc = self.test( X_debiased=X_debiased, g=g)
            print("New parity:", pa)
            # print("Parity of val: " + str(pa))
            # print("Equality of val: " + str(eq))
        return pa, eq, test_f1, val_loss, test_auc

    # Evaluate model
    def test(self, X_debiased, g):
        self.model.eval()
        output = self.model(x=X_debiased, edge_index=g)

        features = self.feature_matrix / self.feature_matrix.norm(dim=0)
        adj_preserve = self.adj_matrix

        if isinstance(self.adj_matrix, torch.Tensor):
            adj_matrix_scipy = self.tensor_to_scipy_sparse(self.adj_matrix)
        else:
            adj_matrix_scipy = self.adj_matrix

        adj = self.sparse_mx_to_torch_sparse_tensor(adj_matrix_scipy)


        adj = adj.to(device)
        features = features.to(device)
        features_preserve = features.clone()
        features_preserve = features_preserve.to(device)
        labels = self.labels.to(device)
        idx_train = self.idx_train.to(device)
        idx_val = self.idx_val.to(device)
        idx_test = self.idx_test.to(device)
        sens = self.sens.to(device)


        print("output:",output)
        # preds = (output.squeeze() > 0).type_as(labels)
        preds = output.argmax(dim=1)

        print("Prediction:", preds)
        # loss_test = F.binary_cross_entropy_with_logits(output[idx_test], labels[idx_test].unsqueeze(1).float())
        # auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu().numpy()], output.detach().cpu().numpy()[idx_test.cpu().numpy()])
        output_np = output.detach().cpu().numpy()
        y_score = output_np[idx_test.cpu().numpy(), 1]
        y_true = labels.cpu().numpy()[idx_test.cpu().numpy()].ravel()
        auc_roc_test = roc_auc_score(y_true, y_score)

        f1_test = f1_score(labels[idx_test.cpu().numpy()].cpu().numpy(), preds[idx_test.cpu().numpy()].cpu().numpy())
        test_auc = auc_roc_test
        test_f1 = f1_test
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "F1_test= {:.4f}".format(test_f1),
        #       "AUC_test= {:.4f}".format(test_auc))
        parity_test, equality_test = fair_metric(preds[idx_test.cpu().numpy()].cpu().numpy(),
                                                labels[idx_test.cpu().numpy()].cpu().numpy(),
                                                sens[idx_test.cpu().numpy()].cpu().numpy())
        print("Parity of test: " + str(parity_test))
        print("Equality of test: " + str(equality_test))
        return parity_test, equality_test, test_f1, test_auc



    def preprosessing(self, epochs=200):

        '''
        Model preprossing part
        '''

        features = self.feature_matrix / self.feature_matrix.norm(dim=0)
        adj_preserve = self.adj_matrix
        
        if isinstance(self.adj_matrix, torch.Tensor):
            adj_matrix_scipy = self.tensor_to_scipy_sparse(self.adj_matrix)
        else:
            adj_matrix_scipy = self.adj_matrix

        adj = self.sparse_mx_to_torch_sparse_tensor(adj_matrix_scipy)

        # adj = self.sparse_mx_to_torch_sparse_tensor(self.adj_matrix)
        model = EDITS(nfeat=features.shape[1], node_num=features.shape[0], nfeat_out=int(features.shape[0]/10), adj_lambda=1e-1, nclass=2, layer_threshold=2, dropout=0.2)  # 3-nba

        model = model.to(device)
        adj = adj.to(device)
        features = features.to(device)
        features_preserve = features.clone()
        features_preserve = features_preserve.to(device)
        labels = self.labels.to(device)
        idx_train = self.idx_train.to(device)
        idx_val = self.idx_val.to(device)
        idx_test = self.idx_test.to(device)
        sens = self.sens.to(device)

        A_debiased, X_debiased = adj, features

        val_adv = []
        test_adv = []
        for epoch in tqdm(range(epochs)):
            if epoch > 400:
                lr = 0.001
            else:
                lr = 1e-5
            model.train()
            model.optimize(adj, features, idx_train, sens, epoch, lr)
            A_debiased, X_debiased, predictor_sens, show, _ = model(adj, features)
            positive_eles = torch.masked_select(predictor_sens[idx_val].squeeze(), sens[idx_val] > 0)
            negative_eles = torch.masked_select(predictor_sens[idx_val].squeeze(), sens[idx_val] <= 0)
            loss_val = - (torch.mean(positive_eles) - torch.mean(negative_eles))
            val_adv.append(loss_val.data)

        param = model.state_dict()

        indices = torch.argsort(param["x_debaising.s"])[:4]
        for i in indices:
            features_preserve[:, i] = torch.zeros_like(features_preserve[:, i])
        X_debiased = features_preserve
        adj1 = sp.csr_matrix(A_debiased.detach().cpu().numpy())
        # print("****************************After debiasing****************************")  # threshold_proportion for GCN: {credit: 0.02, german: 0.25, bail: 0.012}
        # features1 = X_debiased.cpu().float()[:, torch.nonzero(features.sum(axis=0)).squeeze()].detach()
        # if args.dataset != 'german':
        #     features1 = feature_norm(features1)
        # metric_wd(features1, binarize(adj1, adj_preserve, 0.012), sens.cpu(), 0.9, 0)
        # metric_wd(features1, binarize(adj1, adj_preserve, 0.012), sens.cpu(), 0.9, 2)
        # print("****************************************************************************")
        print("Saving into dataset.")
        # sp.save_npz('pre_processed/A_debiased.npz', adj1)
        # torch.save(X_debiased, "pre_processed/X_debiased.pt")
        output_dir = 'pre_processed'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir)

        sp.save_npz(os.path.join(output_dir, 'A_debiased.npz'), adj1)
        torch.save(X_debiased, os.path.join(output_dir, 'X_debiased.pt'))

        print("Preprocessed datasets saved.")


        '''
        Model training and classification part
        '''

    def tensor_to_scipy_sparse(self, tensor):
        """
        Convert a dense PyTorch tensor to a SciPy sparse COO matrix.
        """
        # Ensure the tensor is on CPU and detached from the computation graph.
        if tensor.is_sparse:
            tensor = tensor.to_dense()

        tensor = tensor.cpu().detach().numpy()
        # Optionally, if your tensor is not binary (i.e., representing 0s and 1s),
        # you can convert it accordingly. Here we assume the tensor represents a dense adjacency matrix.
        sparse = sp.coo_matrix(tensor)
        return sparse

    # def evaluate(self):
    #     self.model.eval()
    #     with torch.no_grad():
    #         output_test = self.model(self.feature_matrix, self.adj_norm)
    #         loss_test = F.nll_loss(output_test[self.idx_test], self.labels[self.idx_test])
    #         pred_test = output_test[self.idx_test].max(1)[1]
    #         acc_test  = pred_test.eq(self.labels[self.idx_test]).sum().item() / self.idx_test.size(0)

    #     print(f"Test Loss: {loss_test.item():.4f}, Test Accuracy: {acc_test:.4f}")

    #     """
    #     Fairness Evauation
    #     """

    #     DP = demographic_parity(predictions=pred_test, sens=self.sens[self.idx_test])
    #     EOd = equality_of_odds(predictions=pred_test, labels=self.labels[self.idx_test], sens=self.sens[self.idx_test])
    #     CDP = conditional_demographic_parity(predictions=pred_test, labels=self.labels[self.idx_test], sens=self.sens[self.idx_test])
    #     surrogate = fair_metric(pred_test, self.labels[self.idx_test], self.sens[self.idx_test])
    #     print(f"Demographic Parity: {DP:.4f}, Equality of Odds: {EOd:.4f}, Conditional DP: {CDP:.4f}")
    #     return surrogate, DP.item(), EOd.item(), CDP.item()


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

        print("Matrix updated")
        device = torch.device(f"cuda:{gpu_index}"if torch.cuda.is_available() else "cpu")
        self.adj_matrix = adj_matrix.to(device)
        self.adj_norm = normalize_adjacency(self.adj_matrix).detach()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def binarize(self, A_debiased, adj_ori,
     threshold_proportion):

        the_con1 = (A_debiased - adj_ori).A
        the_con1 = np.where(the_con1 > np.max(the_con1) * threshold_proportion, 1 + the_con1 * 0, the_con1)
        the_con1 = np.where(the_con1 < np.min(the_con1) * threshold_proportion, -1 + the_con1 * 0, the_con1)
        the_con1 = np.where(np.abs(the_con1) == 1, the_con1, the_con1 * 0)
        A_debiased = adj_ori + sp.coo_matrix(the_con1)
        assert A_debiased.max() == 1
        assert A_debiased.min() == 0
        A_debiased = self.normalize_scipy(A_debiased)
        return A_debiased
    
    def normalize_scipy(self, mx):

        if isinstance(mx, torch.Tensor):
            if mx.is_sparse:
                mx = mx.to_dense()
            mx = mx.cpu().detach().numpy()
        
        if not sp.isspmatrix(mx):
            mx = sp.coo_matrix(mx)

        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

# s = victim()
# s.train()
# s.evaluate()