import torch.nn as nn
import torch.nn.functional as F
from layers import GCNLayer

class GCN(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, dropout = 0.5):
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(in_features=in_features, out_features=hidden_features)
        self.gc2 = GCNLayer(in_features=hidden_features, out_features=out_features)
        self.dropout = dropout

    def forward(self, x, adj_norm):
        # First layer
        x = self.gc1(x, adj_norm)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # Second layer
        x = self.gc2(x, adj_norm)
        return F.log_softmax(x, dim=1)
