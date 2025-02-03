import torch.nn as nn
from FairGNN.GCN import GCN,GCN_Body
from FairGNN.GAT import GAT,GAT_body
import torch

def get_model(nfeat, num_hidden, dropout):
    # if args.model == "GCN":
    model = GCN_Body(nfeat,num_hidden,dropout)
    # elif args.model == "GAT":
    #     heads =  ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    #     model = GAT_body(args.num_layers,nfeat,args.num_hidden,heads,args.dropout,args.attn_drop,args.negative_slope,args.residual)
    # else:
    #     print("Model not implement")
    #     return

    return model

class FairGNN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, weight_decay, lr, dropout, alpha, beta):
        super(FairGNN,self).__init__()

        self.estimator = GCN(nfeat = nfeat, nhid = nhid, nclass = 1, dropout = dropout)
        self.GNN = get_model(nfeat,num_hidden=nhid, dropout=dropout)
        self.classifier = nn.Linear(nhid,1)
        self.adv = nn.Linear(nhid,1)
        
        G_params = list(self.GNN.parameters()) + list(self.classifier.parameters()) + list(self.estimator.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr = lr, weight_decay = weight_decay)
        self.optimizer_A = torch.optim.Adam(self.adv.parameters(), lr = lr, weight_decay = weight_decay)

        # self.args = args
        self.criterion = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.beta = beta

        self.G_loss = 0
        self.A_loss = 0

    def reset_parameters(self):
        self.estimator.reset_parameters()
        self.GNN.reset_parameters()
        self.classifier.reset_parameters()
        self.adv.reset_parameters()

    def forward(self,g,x):
        s = self.estimator(g,x)
        z = self.GNN(g,x)
        y = self.classifier(z)
        # print("Result of classifier: ", y)
        # print("Result of estimator: ", s)
        return y,s

    def optimize(self,g,x,labels,idx_train,sens,idx_sens_train):
        self.train()

        self.adv.requires_grad_(False)
        self.optimizer_G.zero_grad()

        s = self.estimator(g,x)
        h = self.GNN(g,x)
        y = self.classifier(h)

        s_g = self.adv(h)

        s_score = torch.sigmoid(s.detach())
        # s_score = (s_score > 0.5).float()
        s_score[idx_sens_train]=sens[idx_sens_train].unsqueeze(1).float()
        y_score = torch.sigmoid(y)
        self.cov =  torch.abs(torch.mean((s_score - torch.mean(s_score)) * (y_score - torch.mean(y_score))))

        self.cls_loss = self.criterion(y[idx_train],labels[idx_train].unsqueeze(1).float())
        self.adv_loss = self.criterion(s_g,s_score)

        if torch.isnan(self.cls_loss) or torch.isnan(self.adv_loss) or torch.isnan(self.cov):
            print("Found NaNs in loss components!")
        # print("cls_loss: ", self.cls_loss)
        # print("adv_loss: ", self.adv_loss)
        # print("cov: ", self.cov)
        self.G_loss = self.cls_loss  + self.alpha * self.cov - self.beta * self.adv_loss
        # print("G_loss: ", self.G_loss)
        self.G_loss.backward()
        self.optimizer_G.step()

        ## update Adv
        self.adv.requires_grad_(True)
        self.optimizer_A.zero_grad()
        s_g = self.adv(h.detach())
        self.A_loss = self.criterion(s_g,s_score)
        self.A_loss.backward()
        self.optimizer_A.step()

