import torch.nn as nn
import torch.nn.functional as F
from .layers import GCNLayer
import torch

gpu_index = 2

class GCN(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, dropout = 0.5):

        super(GCN, self).__init__()
        self.gc1 = GCNLayer(in_features=in_features, out_features=hidden_features)
        self.with_relu = True
        self.gc2 = GCNLayer(in_features=hidden_features, out_features=out_features)
        self.dropout = dropout
        device = torch.device(f"cuda:{gpu_index}"if torch.cuda.is_available() else "cpu")
        # self.gc1.to(device)
        # self.gc2.to(device)
        # self.to(device)
    
    def forward(self, x, adj_norm):
        # First layer
        # print(x.size())
        x = self.gc1(x, adj_norm)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # Second layer
        x = self.gc2(x, adj_norm)
        return x

    def initialization(self):
        self.gc1.reset_parameters()
        # self.gc2.reset_parameters()
        self.W.reset_parameters()
