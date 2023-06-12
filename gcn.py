"""
This code was copied from the GCN implementation in DGL examples.
"""
import torch
import torch.nn as nn

from dgl.nn.pytorch import SGConv
from torch_geometric.nn import GCNConv
class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 bias = True,
                 weight=True):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.bns = torch.nn.ModuleList()

        self.layers.append(GCNConv(in_feats, n_hidden))
        self.bns.append(torch.nn.BatchNorm1d(n_hidden, momentum = 0.01))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNConv(n_hidden, n_hidden))
            # if i != (n_layers - 2):
            self.bns.append(torch.nn.BatchNorm1d(n_hidden, momentum = 0.01))
        # output layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features,edge_index):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h,edge_index)
            # h = self.bns[i](h)
        return h

