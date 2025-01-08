import torch
import torch.nn as nn
import numpy as np


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias = True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj_norm):
        
        print("Type of x: ", type(x))
        if isinstance(x, torch.Tensor) :
            print("Type!!! of x: ", type(x))
            x = x.detach().cpu().numpy()
        xnp = np.vstack(x).astype(np.float32)
        x = torch.from_numpy(xnp)
        # x.detach().cpu().numpy()
        support = torch.matmul(x, self.weight)
        # adj_norm = adj_norm.float()
        print(adj_norm)
        print(support)

        out = torch.matmul(torch.from_numpy(adj_norm).to(torch.float32), support)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'