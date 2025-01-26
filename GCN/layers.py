import torch
import torch.nn as nn
import numpy as np
gpu_index = 2
device = torch.device(f"cuda:{gpu_index}"if torch.cuda.is_available() else "cpu")

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias = True):
        # print("in feature:", in_features)
        # print("out feature:", out_features)
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features),requires_grad=True)
        # self.weight.requires_grad = True
        # self.weight.to(device)
        # print("weight device:", device)
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features),requires_grad=True)
            # self.bias.requires_grad = True
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.to(device)
        param_count = sum(p.numel() for p in self.parameters())
        print(f"GCNLayer initialized on {device}. Number of parameters: {param_count}")


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj_norm):

        adj_norm = torch.from_numpy(adj_norm).to(torch.float32).to(device)

        if isinstance(x, torch.Tensor) :
            x = x.detach().cpu().numpy()  # Ensure it's on CPU before converting to NumPy
        xnp = np.vstack(x).astype(np.float32)
        # device = torch.device(f"cuda:{gpu_index}"if torch.cuda.is_available() else "cpu")

        x = torch.from_numpy(xnp).to(device)
        # x.to(device)

        # print("x device:", device)
        support = torch.matmul(x, self.weight)

        out = torch.matmul(adj_norm, support)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'