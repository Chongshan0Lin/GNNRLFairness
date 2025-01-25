import torch
import torch.nn as nn
import numpy as np
gpu_index = 2

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias = True):
        # print("in feature:", in_features)
        # print("out feature:", out_features)
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        device = torch.device(f"cuda:{gpu_index}"if torch.cuda.is_available() else "cpu")
        self.weight.to(device)
        print("weight device:", device)
        
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

        adj_norm = torch.from_numpy(adj_norm).to(torch.float32)

        if isinstance(x, torch.Tensor) :
            x = x.detach().cpu().numpy()
        xnp = np.vstack(x).astype(np.float32)
        x = torch.from_numpy(xnp)
        device = torch.device(f"cuda:{gpu_index}"if torch.cuda.is_available() else "cpu")
        x.to(device)

        print("x device:", device)
        support = torch.matmul(x, self.weight)

        out = torch.matmul(adj_norm, support)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'