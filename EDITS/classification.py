import time
import argparse
import numpy as np

import torch
print(torch.__version__)

import torch.nn.functional as F
import torch.optim as optim
from metrics import metric_wd
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
import scipy.sparse as sp
# from utils import load_bail, load_credit, load_german, feature_norm, normalize_scipy
from .gcn import GCN
from torch_geometric.utils import dropout_adj, convert
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')


adj_ori = adj
adj = normalize_scipy(adj)



def train(epoch, pa, eq, test_f1, val_loss, test_auc):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(x=X_debiased, edge_index=torch.LongTensor(edge_index.cpu()).cuda())
    preds = (output.squeeze() > 0).type_as(labels)
    loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
    auc_roc_train = roc_auc_score(labels.cpu().numpy()[idx_train.cpu().numpy()], output.detach().cpu().numpy()[idx_train.cpu().numpy()])
    f1_train = f1_score(labels[idx_train.cpu().numpy()].cpu().numpy(), preds[idx_train.cpu().numpy()].cpu().numpy())
    loss_train.backward()
    optimizer.step()
    _, _ = fair_metric(preds[idx_train.cpu().numpy()].cpu().numpy(), labels[idx_train.cpu().numpy()].cpu().numpy(), sens[idx_train.cpu().numpy()].cpu().numpy())

    model.eval()
    output = model(x=X_debiased, edge_index=torch.LongTensor(edge_index.cpu()).cuda())
    preds = (output.squeeze() > 0).type_as(labels)
    loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float())
    auc_roc_val = roc_auc_score(labels.cpu().numpy()[idx_val.cpu().numpy()], output.detach().cpu().numpy()[idx_val.cpu().numpy()])
    f1_val = f1_score(labels[idx_val.cpu().numpy()].cpu().numpy(), preds[idx_val.cpu().numpy()].cpu().numpy())
    # print('Epoch: {:04d}'.format(epoch + 1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'F1_train: {:.4f}'.format(f1_train),
    #       'AUC_train: {:.4f}'.format(auc_roc_train),
    #       'loss_val: {:.4f}'.format(loss_val.item()),
    #       'F1_val: {:.4f}'.format(f1_val),
    #       'AUC_val: {:.4f}'.format(auc_roc_val),
    #       'time: {:.4f}s'.format(time.time() - t))

    if epoch < 15:
        return 0, 0, 0, 1e5, 0
    if loss_val < val_loss:
        val_loss = loss_val.data
        pa, eq, test_f1, test_auc = test(test_f1)
        # print("Parity of val: " + str(pa))
        # print("Equality of val: " + str(eq))
    return pa, eq, test_f1, val_loss, test_auc


def test(test_f1):
    model.eval()
    output = model(x=X_debiased, edge_index=torch.LongTensor(edge_index.cpu()).cuda())
    preds = (output.squeeze() > 0).type_as(labels)
    loss_test = F.binary_cross_entropy_with_logits(output[idx_test], labels[idx_test].unsqueeze(1).float())
    auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu().numpy()], output.detach().cpu().numpy()[idx_test.cpu().numpy()])
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
    # print("Parity of test: " + str(parity_test))
    # print("Equality of test: " + str(equality_test))
    return parity_test, equality_test, test_f1, test_auc


# Train model
t_total = time.time()
val_loss = 1e5
pa = 0
eq = 0
test_auc = 0
test_f1 = 0
for epoch in tqdm(range(args.epochs)):
    pa, eq, test_f1, val_loss, test_auc = train(epoch, pa, eq, test_f1, val_loss, test_auc)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print("Delta_{SP}: " + str(pa))
print("Delta_{EO}: " + str(eq))
print("F1: " + str(test_f1))
print("AUC: " + str(test_auc))
