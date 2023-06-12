import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import dgl.function as fn
from gcn import GCN
class ClassMLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(ClassMLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)
        #return x
class GGD_Encoder(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation, dropout):
        super(GGD_Encoder, self).__init__()
        self.conv = GCN(in_feats, n_hidden, n_hidden, n_layers, activation, dropout)

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(features.shape[0])
            features = features[perm]
        features = self.conv(features)

        return features

class ClassMLP_encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super(ClassMLP_encoder, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x,corrupt=False):
        if corrupt:
            perm = torch.randperm(x.shape[0])
            x = x[perm]
        for i, lin in enumerate(self.lins[:-1]):
            if i!=0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin(x)
            # x = self.bns[i](x)
            # x = F.relu(x)
        x = self.lins[-1](x)
        # return torch.log_softmax(x, dim=-1)
        return x


class GGD(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation, dropout, proj_layers):
        super(GGD, self).__init__()
        self.encoder = GGD_Encoder(in_feats, n_hidden, n_layers, activation, dropout)
        self.mlp = torch.nn.ModuleList()
        for i in range(proj_layers):
            self.mlp.append(nn.Linear(n_hidden, n_hidden))
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, features, labels, loss_func):
        h_1 = self.encoder(features, corrupt=False)
        h_2 = self.encoder(features, corrupt=False)

        sc_1 = h_1.squeeze(0)
        sc_2 = h_2.squeeze(0)
        for i, lin in enumerate(self.mlp):
            sc_1 = lin(sc_1)
            sc_2 = lin(sc_2)

        sc_1 = sc_1.sum(1).unsqueeze(0)
        sc_2 = sc_2.sum(1).unsqueeze(0)
        lbl_1 = torch.ones(1, sc_1.shape[1])
        lbl_2 = torch.zeros(1, sc_1.shape[1])
        lbl = torch.cat((lbl_1, lbl_2), 1).cuda()
        logits = torch.cat((sc_1, sc_2), 1)

        loss = loss_func(logits, lbl)

        return loss

    def embed(self, features, g):
        h_1 = self.encoder(features, corrupt=False)

        feat = h_1.clone().squeeze(0)

        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        norm = norm.to(h_1.device).unsqueeze(1)
        for _ in range(10):
            feat = feat * norm
            g.ndata['h2'] = feat
            g.update_all(fn.copy_u('h2', 'm'),
                             fn.sum('m', 'h2'))
            feat = g.ndata.pop('h2')
            feat = feat * norm

        h_2 = feat.unsqueeze(0)

        return h_1.detach(), h_2.detach()
    
class PGL(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, dropout, proj_layers):
        super(PGL, self).__init__()
        self.encoder = ClassMLP_encoder(in_feats, n_hidden, n_layers, dropout)
        self.mlp = torch.nn.ModuleList()
        for i in range(proj_layers):
            self.mlp.append(nn.Linear(n_hidden, n_hidden))
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, features, features_n, labels, loss_func):
        h_1 = self.encoder(features, corrupt=False)
        h_2 = self.encoder(features_n, corrupt=False)

        sc_1 = h_1.squeeze(0)
        sc_2 = h_2.squeeze(0)
        for i, lin in enumerate(self.mlp):
            sc_1 = lin(sc_1)
            sc_2 = lin(sc_2)

        sc_1 = sc_1.sum(1).unsqueeze(0)
        sc_2 = sc_2.sum(1).unsqueeze(0)
        lbl_1 = torch.ones(1, sc_1.shape[1])
        lbl_2 = torch.zeros(1, sc_2.shape[1])
        lbl = torch.cat((lbl_1, lbl_2), 1).cuda()

        logits = torch.cat((sc_1, sc_2), 1)

        loss = loss_func(logits, lbl)

        return loss

    def embed(self, features):
        h_1 = self.encoder(features, corrupt=False)

        return h_1.detach()
class Classifier(nn.Module):
    def __init__(self, n_hidden, n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(n_hidden, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, features):
        features = self.fc(features)
        return torch.log_softmax(features, dim=-1)