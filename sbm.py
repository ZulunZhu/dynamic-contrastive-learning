import time
import uuid
import random
import argparse
import gc
import torch
import resource
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import *
from model import ClassMLP
from propagation import InstantGNN
import math
import sklearn.preprocessing

import os

import psutil
import sys
import pdb

def train(model, device, train_loader, optimizer, loss_fn, use_pdb=False):
    model.train()

    time_epoch = 0
    loss_list, acc_list = [], []

    for i, (x, y) in enumerate(train_loader):
        t_st = time.time()
        x, y = x.cuda(device), y.cuda(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out, y.squeeze(1))
        b = 0.1
        flood = (loss - b).abs() + b
        acc = com_accuracy(out, y)
        acc_list.append(acc.item())
        if use_pdb:
            pdb.set_trace(header='train')

        flood.backward()
        optimizer.step()
        #time_epoch += (time.time() - t_st)
        loss_list.append(loss.item())
        time_epoch += (time.time() - t_st)
    return np.mean(loss_list), np.mean(acc_list), time_epoch


@torch.no_grad()
def validate(model, device, loader, loss_fn, use_pdb=False):
    model.eval()
    loss_list, acc_list = [], []
    for i, (x, y) in enumerate(loader):
        x, y = x.cuda(device), y.cuda(device)
        out = model(x)

        #loss = loss_fn(out.view(y.shape), y.float())
        loss = F.nll_loss(out, y.squeeze(1))
        loss_list.append(loss.item())
        acc = com_accuracy(out, y)
        acc_list.append(acc.item())
        if use_pdb:
            pdb.set_trace(header='valid')

    return np.mean(loss_list), np.mean(acc_list)

@torch.no_grad()
def test(model, device, loader, checkpt_file, loss_fn, use_pdb=False):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    loss_list, acc_list = [], []
    for step, (x, y) in enumerate(loader):
        x, y = x.cuda(device), y.cuda(device)
        out = model(x)

        #loss = loss_fn(out.view(y.shape), y.float())
        loss = F.nll_loss(out, y.squeeze(1))
        loss_list.append(loss.item())
        acc = com_accuracy(out, y)
        acc_list.append(acc.item())

        if use_pdb:
            pdb.set_trace(header='test')

    return np.mean(loss_list), np.mean(acc_list)

## load feat and generate model
def prepare_to_train(features, train_idx, val_idx, test_idx, train_labels, val_labels, test_labels, loss_fn, args,fineturn=False):
    print(args)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)
    
    features = torch.FloatTensor(features)

    features_train = features[train_idx]
    features_val = features[val_idx]
    features_test = features[test_idx]
    del features
    gc.collect()

    train_dataset = SimpleDataset(features_train, train_labels)
    valid_dataset = SimpleDataset(features_val, val_labels)
    test_dataset = SimpleDataset(features_test, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(val_labels), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_labels), shuffle=False)

    label_dim = int(max(train_labels.max(),val_labels.max(),test_labels.max()))+1
    model = ClassMLP(features_train.size(-1), args.hidden, label_dim, args.layer, args.dropout).cuda(args.dev)
    if fineturn:
        model.load_state_dict(torch.load(args.checkpt_file))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #### begin train
    bad_counter = 0
    best = 0
    best_epoch = 0
    train_time = 0
    best_loss = 1e+8 * 1.0
    model.reset_parameters()
    print("--------------------------")
    print("Training...")
    for epoch in range(args.epochs):
        loss_tra, acc_tra, train_ep = train(model, args.dev, train_loader, optimizer, loss_fn)
        loss_val, acc_val = validate(model, args.dev, valid_loader, loss_fn)
        train_time += train_ep
        if (epoch + 1) % 2 == 0:
            print(f'Epoch:{epoch + 1:02d},'
                  f'Train_loss:{loss_tra:.8f}',
                  f'Train_acc:{acc_tra:.5f}',
                  f'Valid_loss:{loss_val:.8f}',
                  f'Valid_acc:{acc_val:.5f}',
                  f'Time_cost:{train_ep:.3f} / {train_time:.3f}')
        if acc_val > best:
            best = acc_val
            best_epoch = epoch + 1
            torch.save(model.state_dict(), args.checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter == args.patience:
            break

    loss_test, acc_test = test(model, args.dev, test_loader, args.checkpt_file, loss_fn)
    print('Load {}th epoch'.format(best_epoch))
    print(f"Test loss:{loss_test:.8f}, acc:{acc_test:.5f}")

def main():
    mem = psutil.virtual_memory()
    initial_memory = mem.used / 1024 / 1024 / 1024
    parser = argparse.ArgumentParser()
    # Dataset and Algorithom
    parser.add_argument('--seed', type=int, default=20159, help='random seed.')
    parser.add_argument('--dataset', default='SBM-50000-50-20+1', help='dateset.')
    # Algorithm parameters
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha for APPNP_AGP.')
    parser.add_argument('--rmax', type=float, default=1e-7, help='threshold.')
    # Learining parameters
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay.')
    parser.add_argument('--layer', type=int, default=2, help='number of layers.')
    parser.add_argument('--hidden', type=int, default=256, help='hidden dimensions.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate.')
    parser.add_argument('--bias', default='none', help='bias.')
    parser.add_argument('--epochs', type=int, default=3000, help='number of epochs.')
    parser.add_argument('--batch', type=int, default=1024, help='batch size.')
    parser.add_argument('--patience', type=int, default=20, help='patience.')
    parser.add_argument('--dev', type=int, default=1, help='device id.')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print("--------------------------")
    print(args)
    args.checkpt_file = 'pretrained/' + uuid.uuid4().hex + '.pt'

    features, train_labels, val_labels, test_labels, train_idx, val_idx, test_idx, memory_dataset, py_alg= load_sbm_init(args.dataset, args.rmax, args.alpha)
    mem = psutil.virtual_memory()
    memory_cost = mem.used / 1024 / 1024 / 1024 - initial_memory
    print('load_init cost: ', memory_cost)

    loss_fn = torch.nn.CrossEntropyLoss()
    prepare_to_train(features, train_idx, val_idx, test_idx, train_labels, val_labels, test_labels, loss_fn, args)

    print('--------------------- update ----------------------')
    for i in range(10):
        py_alg.snapshot_operation('../data/'+args.dataset+'/'+args.dataset+'_Edgeupdate_snap'+str(i)+'.txt', args.rmax, args.alpha, features)
        mem = psutil.virtual_memory()
        memory_cost = mem.used / 1024 / 1024 / 1024 - initial_memory
        print('snapshot_operation ' + str(i) + ' cost: ', memory_cost)
        data = np.loadtxt('./data/'+args.dataset+'/'+args.dataset+'_label_snap'+str(i)+'.txt')
        train_labels = torch.LongTensor(data[train_idx])
        val_labels = torch.LongTensor(data[val_idx])
        test_labels = torch.LongTensor(data[test_idx])
        train_labels = train_labels.reshape(train_labels.size(0), 1)
        val_labels = val_labels.reshape(val_labels.size(0), 1)
        test_labels = test_labels.reshape(test_labels.size(0), 1)

        prepare_to_train(features, train_idx, val_idx, test_idx, train_labels, val_labels, test_labels, loss_fn, args)

if __name__ == '__main__':
    main()


