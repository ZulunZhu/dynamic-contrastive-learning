from ogb.nodeproppred import PygNodePropPredDataset
import argparse
from tqdm import tqdm
import numpy as np
import random
import scipy.sparse as sp
import torch
import torch.nn.functional as F
# from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import sklearn.preprocessing
# import tracemalloc
import gc
import struct
# from torch_sparse import coalesce
import math
# import pdb
import scipy
import time
import dataset

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
print('start processing data: ')   
self_loop = True
root_folder = "/home/ubuntu/project"
# root_folder = ".."
def dropout_adj(edge_index, rmnode_idx, edge_attr=None, force_undirected=True,
                num_nodes=None):

    N = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    row, col = edge_index
    
    if force_undirected:
        row, col, edge_attr = filter_adj(row, col, edge_attr, row < col)
    convert_start = time.time()
    # row_convert = row.numpy().tolist()
    # col_convert = col.numpy().tolist()
    convert_end = time.time()
    print('convert cost:', convert_end - convert_start)

    row_mask = np.isin(row, rmnode_idx)
    col_mask = np.isin(col, rmnode_idx)
    drop_mask = torch.from_numpy(np.logical_or(row_mask, col_mask)).to(torch.bool)

    mask = ~drop_mask

    new_row, new_col, edge_attr = filter_adj(row, col, edge_attr, mask)
    drop_row, drop_col, edge_attr = filter_adj(row, col, edge_attr, drop_mask)
    print('init:',len(new_row), ', drop:', len(drop_row))

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([new_row, new_col], dim=0),
             torch.cat([new_col, new_row], dim=0)], dim=0)
        # if edge_attr is not None:
        #     edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        # edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
    else:
        edge_index = torch.stack([new_row, new_col], dim=0)
    drop_edge_index = torch.stack([drop_row, drop_col], dim=0)  ### only u->v (no v->u)

    return edge_index, drop_edge_index, edge_attr

def filter_adj(row, col, edge_attr, mask):
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]

def arxiv():
    print('start processing data: ')
    dataset=PygNodePropPredDataset(name='ogbn-arxiv',root ="/Resource/dataset/OGB")
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    all_idx = torch.cat([train_idx, val_idx, test_idx])
    print("data:",type(data.x))
    feat=data.x.numpy()
    feat=np.array(feat,dtype=np.float64)
    
    
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)
    print("feat", feat)
    print("feat.shape:", feat.shape)
    print("type(features):", type(feat),feat.dtype)
    # exit(0)
    # np.save(root_folder+'/data/arxiv/arxiv_feat.npy',feat)
    
    #get labels
    labels=data.y
    train_labels=labels.data[train_idx]
    val_labels=labels.data[val_idx]
    test_labels=labels.data[test_idx]
  

    
    train_idx=train_idx.numpy()
    val_idx=val_idx.numpy()
    test_idx=test_idx.numpy()
    train_idx=np.array(train_idx, dtype=np.int32)
    val_idx=np.array(val_idx,dtype=np.int32)
    test_idx=np.array(test_idx,dtype=np.int32)

    train_labels=train_labels.numpy().T
    val_labels=val_labels.numpy().T
    test_labels=test_labels.numpy().T

    train_labels=np.array(train_labels,dtype=np.int32)
    val_labels=np.array(val_labels,dtype=np.int32)
    test_labels=np.array(test_labels,dtype=np.int32)
    train_labels=train_labels.reshape(train_labels.shape[1])
    val_labels=val_labels.reshape(val_labels.shape[1])
    test_labels=test_labels.reshape(test_labels.shape[1])
    # np.savez(root_folder+'/data/arxiv/arxiv_labels.npz',train_idx=train_idx,val_idx=val_idx,test_idx=test_idx,train_labels=train_labels,val_labels=val_labels,test_labels=test_labels)
    
    print("Edge number before to_undirected:", data.edge_index.size())
    print(type(data.edge_index),data.edge_index)

    data.edge_index=to_undirected(edge_index = data.edge_index,num_nodes = data.num_nodes)
    print("Edge number after to_undirected:", data.edge_index.size())
    row_ful,col_ful=data.edge_index
    row_ful=row_ful.numpy()
    col_ful=col_ful.numpy()
    edge_number = 0
    with open(root_folder+'/data/arxiv/arxiv_full_adj' + '.txt', 'w') as f:
        # if(self_loop):
        for i, j in zip(row_ful, col_ful):
            f.write("%d %d\n" % (i, j))
            f.write("%d %d\n" % (j, i))
            edge_number+=1
        # else:
        #     for i, j in zip(row_ful, col_ful):
        #         if(i != j):
        #             f.write("%d %d\n" % (i, j))
        #             f.write("%d %d\n" % (j, i))
        #             edge_number+=1
    print("full edge_number", edge_number)
    save_adj(row_ful, col_ful, N=data.num_nodes, dataset_name='arxiv', savename='arxiv_full', snap='init')
    

    data.edge_index, drop_edge_index, _ = dropout_adj(data.edge_index,train_idx, num_nodes= data.num_nodes)
    data.edge_index=to_undirected(edge_index = data.edge_index,num_nodes = data.num_nodes)
    
    row_drop, col_drop = np.array(drop_edge_index)

    f = open(root_folder+'/data/arxiv/ogbn-arxiv_update_full.txt', 'w+')
    for k in range(row_drop.shape[0]):
        v_from = row_drop[k]
        v_to = col_drop[k]
        f.write('%d %d\n' % (v_from, v_to))
        f.write('%d %d\n' % (v_to, v_from))
    f.close()
    
    row,col=data.edge_index
    print(row_drop.shape)
    row=row.numpy()
    col=col.numpy()
    edge_number = 0
    with open(root_folder+'/data/arxiv/arxiv_init_adj' + '.txt', 'w') as f:
        if(self_loop):
            for i, j in zip(row, col):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
                edge_number+=1
        else:
            for i, j in zip(row, col):
                if(i != j):
                    f.write("%d %d\n" % (i, j))
                    f.write("%d %d\n" % (j, i))
                    edge_number+=1
    print("edge_number", edge_number)            
                
    torch.save([row, col], root_folder+'/data/arxiv_python/arxiv_init_adj.pt')
    save_adj(row, col, N=data.num_nodes, dataset_name='arxiv', savename='arxiv_init', snap='init')
    num_snap = 16
    snapshot = math.floor(row_drop.shape[0] / num_snap)
    

    #Shuffle the removed edges and show a more flexible setting
    # Generate shuffled indices
    indices = np.random.permutation(len(row_drop))
    # Shuffle both arrays using the same indices
    row_drop = row_drop[indices]
    col_drop = col_drop[indices]

    print('num_snap: ', row_drop)

    
    for sn in range(num_snap):
        print(sn)
        row_sn = row_drop[ sn*snapshot : (sn+1)*snapshot ]
        col_sn = col_drop[ sn*snapshot : (sn+1)*snapshot ]
        if sn == 0:
            row_tmp=row
            col_tmp=col
        
        row_tmp=np.concatenate((row_tmp,row_sn))
        col_tmp=np.concatenate((col_tmp,col_sn))
        row_tmp=np.concatenate((row_tmp,col_sn))
        col_tmp=np.concatenate((col_tmp,row_sn))
        if (sn+1) % 20 ==0 or (sn+1)==num_snap:
            save_adj(row_tmp, col_tmp, N=data.num_nodes, dataset_name='arxiv', savename='arxiv_snap'+str(sn+1), snap=(sn+1)) 
        
        # torch.save([row_sn, col_sn], root_folder+'/data/arxiv_python/arxiv_Edgeupdate_snap' + str(sn+1) + '.pt')

        with open(root_folder+'/data/arxiv/arxiv_Edgeupdate_snap' + str(sn+1+32) + '.txt', 'w') as f:
            if(self_loop):
                for i, j in zip(row_sn, col_sn):
                    f.write("%d %d\n" % (i, j))
                    f.write("%d %d\n" % (j, i))
            else:
                for i, j in zip(row_sn, col_sn):
                    if (i != j):
                        f.write("%d %d\n" % (i, j))
                        f.write("%d %d\n" % (j, i))
    print('Arxiv -- save snapshots finish')




def products():
    dataset=PygNodePropPredDataset(name='ogbn-products',root="../dataset/")
    # dataset=PygNodePropPredDataset(name='ogbn-products',root ="/Resource/dataset/OGB")
    # dataset=PygNodePropPredDataset(name='ogbn-products')
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    all_idx = torch.cat([train_idx, val_idx, test_idx])
    
    #save feat
    feat=data.x.numpy()
    feat=np.array(feat,dtype=np.float64)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)
    print(feat)
    # np.save(root_folder+'/data/products/products_feat.npy',feat)

    #get labels
    print("save labels.....")
    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    
    labels=data.y
    train_labels=labels.data[train_idx]
    val_labels=labels.data[val_idx]
    test_labels=labels.data[test_idx]

    train_idx=train_idx.numpy()
    val_idx=val_idx.numpy()
    test_idx=test_idx.numpy()
    train_idx=np.array(train_idx, dtype=np.int32)
    val_idx=np.array(val_idx,dtype=np.int32)
    test_idx=np.array(test_idx,dtype=np.int32)

    train_labels=train_labels.numpy().T
    val_labels=val_labels.numpy().T
    test_labels=test_labels.numpy().T

    train_labels=np.array(train_labels,dtype=np.int32)
    val_labels=np.array(val_labels,dtype=np.int32)
    test_labels=np.array(test_labels,dtype=np.int32)
    train_labels=train_labels.reshape(train_labels.shape[1])
    val_labels=val_labels.reshape(val_labels.shape[1])
    test_labels=test_labels.reshape(test_labels.shape[1])
    # np.savez(root_folder+'/data/products/products_labels.npz',train_idx=train_idx,val_idx=val_idx,test_idx=test_idx,train_labels=train_labels,val_labels=val_labels,test_labels=test_labels)
    
    data.edge_index=to_undirected(edge_index = data.edge_index,num_nodes = data.num_nodes)

    row_ful,col_ful=data.edge_index
    row_ful=row_ful.numpy()
    col_ful=col_ful.numpy()
    edge_number = 0

    save_adj(row_ful, col_ful, N=data.num_nodes, dataset_name='products', savename='products_full', snap='init')
   
    data.edge_index, drop_edge_index, _ = dropout_adj(data.edge_index,train_idx, num_nodes= data.num_nodes)
    
    shuffle_index=torch.randperm(drop_edge_index.shape[0])

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    
    row_drop, col_drop = np.array(drop_edge_index)
    print('row_drop.shape:', row_drop.shape)
    f = open(root_folder+'/data/products/ogbn-products_update_full.txt', 'w+')
    for k in range(row_drop.shape[0]):
        v_from = row_drop[k]
        v_to = col_drop[k]
        f.write('%d %d\n' % (v_from, v_to))
        f.write('%d %d\n' % (v_to, v_from))
    f.close()
    
    row,col=data.edge_index
    row=row.numpy()
    col=col.numpy()
    edge_number = 0
    with open(root_folder+'/data/products/products_init_adj' + '.txt', 'w') as f:
        if(self_loop):
            for i, j in zip(row, col):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
                edge_number+=1
        else:
            for i, j in zip(row, col):
                if(i != j):
                    f.write("%d %d\n" % (i, j))
                    f.write("%d %d\n" % (j, i))
                    edge_number+=1
    print("edge_number", edge_number) 


    save_adj(row, col, N=data.num_nodes, dataset_name='products', savename='products_init', snap='init')
    num_snap = 32
    snapshot = math.floor(row_drop.shape[0] / num_snap)
    print('num_snap: ', num_snap)

    for sn in range(num_snap):
        print(sn)
        row_sn = row_drop[ 0*snapshot : (sn+1)*snapshot ]
        col_sn = col_drop[ 0*snapshot : (sn+1)*snapshot ]
        if sn == 0:
            row_tmp=row
            col_tmp=col
        
        row_tmp=np.concatenate((row_tmp,row_sn))
        col_tmp=np.concatenate((col_tmp,col_sn))
        row_tmp=np.concatenate((row_tmp,col_sn))
        col_tmp=np.concatenate((col_tmp,row_sn))
        
        save_adj(row_tmp, col_tmp, N=data.num_nodes, dataset_name='products', savename='products_cumusnap'+str(sn+1), snap=(sn+1))
        
        with open(root_folder+'/data/products/products_Edgeupdate_cumusnap' + str(sn+1) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print('Products -- save snapshots finish')

def papers100M_edge():
    s_time = time.time()
    dataset=PygNodePropPredDataset("ogbn-papers100M",root ="/Resource/dataset/OGB")
    print("data is read")
    
    # dataset=PygNodePropPredDataset("ogbn-papers100M")
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    # print("data:",data)
    # feat=data.x.numpy()
    # feat=np.array(feat,dtype=np.float32)

    # #normalize feats
    # scaler = sklearn.preprocessing.StandardScaler()
    # scaler.fit(feat)
    # feat = scaler.transform(feat)

    # #save feats
    # # np.save(root_folder+'/data/papers100M/papers100M_feat.npy',feat)
    # del feat
    # gc.collect()
    print("feature saved")
    # #get labels
    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

    new_edge_index = data.edge_index
    new_num_nodes = data.num_nodes
    del data
    gc.collect()
    # labels=data.y
    # train_labels=labels.data[train_idx]
    # val_labels=labels.data[val_idx]
    # test_labels=labels.data[test_idx]

    # train_idx=train_idx.numpy()
    # val_idx=val_idx.numpy()
    # test_idx=test_idx.numpy()
    # train_idx=np.array(train_idx, dtype=np.int32)
    # val_idx=np.array(val_idx,dtype=np.int32)
    # test_idx=np.array(test_idx,dtype=np.int32)

    # train_labels=train_labels.numpy().T
    # val_labels=val_labels.numpy().T
    # test_labels=test_labels.numpy().T

    # train_labels=np.array(train_labels,dtype=np.int32)
    # val_labels=np.array(val_labels,dtype=np.int32)
    # test_labels=np.array(test_labels,dtype=np.int32)
    # train_labels=train_labels.reshape(train_labels.shape[1])
    # val_labels=val_labels.reshape(val_labels.shape[1])
    # test_labels=test_labels.reshape(test_labels.shape[1])
    # np.savez(root_folder+'/data/papers100M/papers100M_labels.npz',train_idx=train_idx,val_idx=val_idx,test_idx=test_idx,train_labels=train_labels,val_labels=val_labels,test_labels=test_labels)

    print('making the graph undirected')
    print(type(new_edge_index),new_edge_index)
    new_edge_index=to_undirected(edge_index = new_edge_index,num_nodes = new_num_nodes)
    print("process finished cost:", time.time() - s_time)
    
    new_edge_index, drop_edge_index, _ = dropout_adj(new_edge_index, train_idx, force_undirected=False, num_nodes= new_num_nodes)
    print(111111111)
    # data.edge_index = to_undirected(edge_index = data.edge_index,num_nodes = data.num_nodes)
    
    row_drop, col_drop = np.array(drop_edge_index)
    row,col=new_edge_index
    row=row.numpy()
    col=col.numpy()

    edge_number = 0
    with open(root_folder+'/data/papers100M/papers100M_init_adj' + '.txt', 'w') as f:
        if(self_loop):
            for i, j in zip(row, col):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
                edge_number+=1
        else:
            for i, j in zip(row, col):
                if(i != j):
                    f.write("%d %d\n" % (i, j))
                    f.write("%d %d\n" % (j, i))
                    edge_number+=1
    print("edge_number", edge_number)
    
    save_adj(row, col, N=new_num_nodes, dataset_name='papers100M', savename='papers100M_init', snap='init')

    num_snap = 20
    print('num_snap: ',num_snap)
    snapshot = math.floor(row_drop.shape[0] / num_snap)

    for sn in range(num_snap):
        st=sn+1
        print('snap:', st)

        row_sn = row_drop[ sn*snapshot : st*snapshot ]
        col_sn = col_drop[ sn*snapshot : st*snapshot ]
        if sn == 0:
            row_tmp=row
            col_tmp=col
        row_tmp=np.concatenate((row_tmp,row_sn))
        col_tmp=np.concatenate((col_tmp,col_sn))
        row_tmp=np.concatenate((row_tmp,col_sn))
        col_tmp=np.concatenate((col_tmp,row_sn))

        save_adj(row_tmp, col_tmp, N=new_num_nodes, dataset_name='papers100M', savename='papers100M_snap'+str(st), snap=st)

        with open(root_folder+'/data/papers100M/papers100M_Edgeupdate_snap' + str(st) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
        del row_sn,col_sn,row,col
        gc.collect()
    print('Papers100M -- save snapshots finish')

def papers100M_feat_label():
    s_time = time.time()
    dataset=PygNodePropPredDataset("ogbn-papers100M",root ="/Resource/dataset/OGB")
    print("data is read")
    
    # dataset=PygNodePropPredDataset("ogbn-papers100M")
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    # print("data:",data)
    # feat=data.x.numpy()
    # feat=np.array(feat,dtype=np.float32)

    # #normalize feats
    # scaler = sklearn.preprocessing.StandardScaler()
    # scaler.fit(feat)
    # feat = scaler.transform(feat)

    # #save feats
    # # np.save(root_folder+'/data/papers100M/papers100M_feat.npy',feat)
    # del feat
    # gc.collect()
    print("feature saved")
    # #get labels
    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

    
    # labels=data.y
    # train_labels=labels.data[train_idx]
    # val_labels=labels.data[val_idx]
    # test_labels=labels.data[test_idx]

    # train_idx=train_idx.numpy()
    # val_idx=val_idx.numpy()
    # test_idx=test_idx.numpy()
    # train_idx=np.array(train_idx, dtype=np.int32)
    # val_idx=np.array(val_idx,dtype=np.int32)
    # test_idx=np.array(test_idx,dtype=np.int32)

    # train_labels=train_labels.numpy().T
    # val_labels=val_labels.numpy().T
    # test_labels=test_labels.numpy().T

    # train_labels=np.array(train_labels,dtype=np.int32)
    # val_labels=np.array(val_labels,dtype=np.int32)
    # test_labels=np.array(test_labels,dtype=np.int32)
    # train_labels=train_labels.reshape(train_labels.shape[1])
    # val_labels=val_labels.reshape(val_labels.shape[1])
    # test_labels=test_labels.reshape(test_labels.shape[1])
    # np.savez(root_folder+'/data/papers100M/papers100M_labels.npz',train_idx=train_idx,val_idx=val_idx,test_idx=test_idx,train_labels=train_labels,val_labels=val_labels,test_labels=test_labels)

    print('making the graph undirected')
    print(type(data.edge_index),data.edge_index)
    data.edge_index=to_undirected(edge_index = data.edge_index,num_nodes = data.num_nodes)
    print("process finished cost:", time.time() - s_time)
    
    data.edge_index, drop_edge_index, _ = dropout_adj(data.edge_index, train_idx, force_undirected=False, num_nodes= data.num_nodes)
    print(111111111)
    # data.edge_index = to_undirected(edge_index = data.edge_index,num_nodes = data.num_nodes)
    
    row_drop, col_drop = np.array(drop_edge_index)
    row,col=data.edge_index
    save_adj(row, col, N=data.num_nodes, dataset_name='papers100M', savename='papers100M_init', snap='init')
    row=row.numpy()
    col=col.numpy()
    num_snap = 20
    print('num_snap: ',num_snap)
    snapshot = math.floor(row_drop.shape[0] / num_snap)

    for sn in range(num_snap):
        st=sn+1
        print('snap:', st)

        row_sn = row_drop[ sn*snapshot : st*snapshot ]
        col_sn = col_drop[ sn*snapshot : st*snapshot ]
        if sn == 0:
            row_tmp=row
            col_tmp=col
        row_tmp=np.concatenate((row_tmp,row_sn))
        col_tmp=np.concatenate((col_tmp,col_sn))
        row_tmp=np.concatenate((row_tmp,col_sn))
        col_tmp=np.concatenate((col_tmp,row_sn))

        save_adj(row_tmp, col_tmp, N=data.num_nodes, dataset_name='papers100M', savename='papers100M_snap'+str(st), snap=st)

        with open(root_folder+'/data/papers100M/papers100M_Edgeupdate_snap' + str(st) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print('Papers100M -- save snapshots finish')



def tmall():
    # dataset=PygNodePropPredDataset(name='ogbn-products',root="../data/")
    data = dataset.Tmall()
    # data.split_nodes(train_size=0.7, val_size=0.1,
    #              test_size=0.2, random_state=42)
    
    
    # train_idx, val_idx, test_idx = data.train_nodes, data.val_nodes, data.test_nodes
    # # all_idx = torch.cat([train_idx, val_idx, test_idx])
    
    # print("train_idx", train_idx)

    

    # print("feat,",feat)
    # np.save(root_folder+'/data/tmall/tmall_feat.npy',feat)
    feat = torch.FloatTensor(np.load(root_folder+'/data/tmall/tmall_feat.npy'))
    feat=feat[-1]
    feat=np.array(feat,dtype=np.float64)
    print("feat", feat)
    # scaler = sklearn.preprocessing.StandardScaler()
    # scaler.fit(feat)
    # feat = scaler.transform(feat)
    print(feat.shape)
    print(data.edge_index)
    np.save(root_folder+'/data/tmall/tmall_feat.npy',feat)
    exit(0)

    #get labels
    print("save labels.....")
    data.y = data.y.unsqueeze(1)
    
    print("data.y", data.y.shape)

    labels=data.y
    train_labels=labels.data[train_idx]
    val_labels=labels.data[val_idx]
    test_labels=labels.data[test_idx]

    train_idx=train_idx.numpy()
    val_idx=val_idx.numpy()
    test_idx=test_idx.numpy()
    train_idx=np.array(train_idx, dtype=np.int32)
    val_idx=np.array(val_idx,dtype=np.int32)
    test_idx=np.array(test_idx,dtype=np.int32)

    train_labels=train_labels.numpy().T
    val_labels=val_labels.numpy().T
    test_labels=test_labels.numpy().T

    train_labels=np.array(train_labels,dtype=np.int32)
    val_labels=np.array(val_labels,dtype=np.int32)
    test_labels=np.array(test_labels,dtype=np.int32)
    train_labels=train_labels.reshape(train_labels.shape[1])
    val_labels=val_labels.reshape(val_labels.shape[1])
    test_labels=test_labels.reshape(test_labels.shape[1])
    
    # np.savez(root_folder+'/data/tmall/tmall_labels.npz',train_idx=train_idx,val_idx=val_idx,test_idx=test_idx,train_labels=train_labels,val_labels=val_labels,test_labels=test_labels)
    
    # print("data_ogb.edge_index", data_ogb.edge_index.shape, data_ogb.edge_index)
    data.edge_index = torch.tensor(data.edge_index).squeeze(0)
    print("data.edges", data.edge_index.shape, data.edge_index)

    print("Edge number before to_undirected:", data.edge_index.size())
    # print(type(data_ogb.edge_index),data_ogb.edge_index.size())
    
    
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    
    print("Edge number after to_undirected:", data.edge_index.size())
    

    data.edge_index, drop_edge_index, _ = dropout_adj(data.edge_index,train_idx, num_nodes= data.num_nodes)
    
    # data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    print("Edge number after to_undirected:", data.edge_index.size())

    row_drop, col_drop = np.array(drop_edge_index)
    print('row_drop.shape:', row_drop.shape)
    f = open(root_folder+'/data/tmall/tmall_update_full.txt', 'w+')
    for k in range(row_drop.shape[0]):
        v_from = row_drop[k]
        v_to = col_drop[k]
        f.write('%d %d\n' % (v_from, v_to))
        f.write('%d %d\n' % (v_to, v_from))
    f.close()
    
    row,col=data.edge_index
    row=row.numpy()
    col=col.numpy()
    edge_number = 0
    with open(root_folder+'/data/tmall/tmall_init_adj' + '.txt', 'w') as f:
        if(self_loop):
            for i, j in zip(row, col):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
                edge_number+=1
        else:
            for i, j in zip(row, col):
                if(i != j):
                    f.write("%d %d\n" % (i, j))
                    f.write("%d %d\n" % (j, i))
                    edge_number+=1
    print("edge_number", edge_number) 
    exit(0)
    save_adj(row, col, N=data.num_nodes, dataset_name='tmall', savename='tmall_init', snap='init')
    num_snap = 20
    snapshot = math.floor(row_drop.shape[0] / num_snap)
    print('num_snap: ', num_snap)

    for sn in range(num_snap):
        print(sn)
        row_sn = row_drop[ sn*snapshot : (sn+1)*snapshot ]
        col_sn = col_drop[ sn*snapshot : (sn+1)*snapshot ]
        if sn == 0:
            row_tmp=row
            col_tmp=col
        
        row_tmp=np.concatenate((row_tmp,row_sn))
        col_tmp=np.concatenate((col_tmp,col_sn))
        row_tmp=np.concatenate((row_tmp,col_sn))
        col_tmp=np.concatenate((col_tmp,row_sn))
        
        save_adj(row_tmp, col_tmp, N=data.num_nodes, dataset_name='tmall', savename='tmall_snap'+str(sn+1), snap=(sn+1))
        
        with open(root_folder+'/data/tmall/tmall_Edgeupdate_snap' + str(sn+1) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print('tmall -- save snapshots finish')

def patent():
    # dataset=PygNodePropPredDataset(name='ogbn-products',root="../data/")
    # dataset_ogb=PygNodePropPredDataset(name='ogbn-arxiv',root ="/Resource/dataset/OGB")
    # data_ogb = dataset_ogb[0]


    data = dataset.Patent()
    data.split_nodes(train_size=0.7, val_size=0.1,
                 test_size=0.2, random_state=42)
    
    
    train_idx, val_idx, test_idx = data.train_nodes, data.val_nodes, data.test_nodes
    # all_idx = torch.cat([train_idx, val_idx, test_idx])
    
    print("train_idx", train_idx)
    
    
    # exit(0)
    #get labels
    print("save labels.....")
    data.y = data.y.unsqueeze(1)
    
    print("data.y", data.y.shape)

    labels=data.y
    train_labels=labels.data[train_idx]
    val_labels=labels.data[val_idx]
    test_labels=labels.data[test_idx]

    train_idx=train_idx.numpy()
    val_idx=val_idx.numpy()
    test_idx=test_idx.numpy()
    train_idx=np.array(train_idx, dtype=np.int32)
    val_idx=np.array(val_idx,dtype=np.int32)
    test_idx=np.array(test_idx,dtype=np.int32)

    train_labels=train_labels.numpy().T
    val_labels=val_labels.numpy().T
    test_labels=test_labels.numpy().T

    train_labels=np.array(train_labels,dtype=np.int32)
    val_labels=np.array(val_labels,dtype=np.int32)
    test_labels=np.array(test_labels,dtype=np.int32)
    train_labels=train_labels.reshape(train_labels.shape[1])
    val_labels=val_labels.reshape(val_labels.shape[1])
    test_labels=test_labels.reshape(test_labels.shape[1])
    
    np.savez(root_folder+'/data/patent/patent_labels.npz',train_idx=train_idx,val_idx=val_idx,test_idx=test_idx,train_labels=train_labels,val_labels=val_labels,test_labels=test_labels)
    
    # print("data_ogb.edge_index", data_ogb.edge_index.shape, data_ogb.edge_index)
    data.edge_index = torch.tensor(data.edge_index).squeeze(0)
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row_ful,col_ful=data.edge_index
    row_ful=row_ful.numpy()
    col_ful=col_ful.numpy()
    edge_number = 0

    save_adj(row_ful, col_ful, N=data.num_nodes, dataset_name='patent', savename='patent_full', snap='init')
   
    data.edge_index, drop_edge_index, _ = dropout_adj(data.edge_index,train_idx, num_nodes= data.num_nodes)
    
    shuffle_index=torch.randperm(drop_edge_index.shape[0])

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    
    row_drop, col_drop = np.array(drop_edge_index)
    print('row_drop.shape:', row_drop.shape)
    f = open(root_folder+'/data/patent/patent_update_full.txt', 'w+')
    for k in range(row_drop.shape[0]):
        v_from = row_drop[k]
        v_to = col_drop[k]
        f.write('%d %d\n' % (v_from, v_to))
        f.write('%d %d\n' % (v_to, v_from))
    f.close()
    
    row,col=data.edge_index
    row=row.numpy()
    col=col.numpy()
    edge_number = 0
    with open(root_folder+'/data/patent/patent_init_adj' + '.txt', 'w') as f:
        if(self_loop):
            for i, j in zip(row, col):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
                edge_number+=1
        else:
            for i, j in zip(row, col):
                if(i != j):
                    f.write("%d %d\n" % (i, j))
                    f.write("%d %d\n" % (j, i))
                    edge_number+=1
    print("edge_number", edge_number) 


    save_adj(row, col, N=data.num_nodes, dataset_name='patent', savename='patent_init', snap='init')
    num_snap = 17
    snapshot = math.floor(row_drop.shape[0] / num_snap)
    print('num_snap: ', num_snap)

    for sn in range(num_snap):
        print(sn)
        row_sn = row_drop[ sn*snapshot : (sn+1)*snapshot ]
        col_sn = col_drop[ sn*snapshot : (sn+1)*snapshot ]
        if sn == 0:
            row_tmp=row
            col_tmp=col
        
        row_tmp=np.concatenate((row_tmp,row_sn))
        col_tmp=np.concatenate((col_tmp,col_sn))
        row_tmp=np.concatenate((row_tmp,col_sn))
        col_tmp=np.concatenate((col_tmp,row_sn))
        
        save_adj(row_tmp, col_tmp, N=data.num_nodes, dataset_name='patent', savename='patent_snap'+str(sn+1), snap=(sn+1))
        
        with open(root_folder+'/data/patent/patent_Edgeupdate_snap' + str(sn+1) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print('patent -- save snapshots finish')

def mooc():
    # dataset=PygNodePropPredDataset(name='ogbn-products',root="../data/")
    # dataset_ogb=PygNodePropPredDataset(name='ogbn-arxiv',root ="/Resource/dataset/OGB")
    # data_ogb = dataset_ogb[0]


    data = dataset.mooc()
    # data.split_nodes(train_size=0.7, val_size=0.1,
    #              test_size=0.2, random_state=42)
    
    
    
    # train_idx, val_idx, test_idx = data.train_nodes, data.val_nodes, data.test_nodes
    # # all_idx = torch.cat([train_idx, val_idx, test_idx])

    train_idx = np.load(root_folder+'/dataset/mooc/train_mooc.npy')
    val_idx = np.load(root_folder+'/dataset/mooc/val_mooc.npy')
    test_idx = np.load(root_folder+'/dataset/mooc/test_mooc.npy')
    all_size = np.arange(data.num_nodes)
    # train_idx = all_size[train_idx]
    # val_idx = all_size[val_idx]
    # test_idx = all_size[test_idx]

    print("new_train_idx",train_idx.shape)
    print("new_train_idx",val_idx.shape)
    print("new_train_idx",test_idx.shape)
    # exit(0)
    
    # exit(0)
    #get labels
    print("save labels.....")
    data.y = data.y.unsqueeze(1)
    
    print("data.y", data.y.shape)

    labels=data.y
    train_labels=labels.data[train_idx]
    val_labels=labels.data[val_idx]
    test_labels=labels.data[test_idx]

    print("np.sum(train_labels)", torch.sum(train_labels))
    # exit(0)
    # train_idx=train_idx.numpy()
    # val_idx=val_idx.numpy()
    # test_idx=test_idx.numpy()
    train_idx=np.array(train_idx, dtype=np.int32)
    val_idx=np.array(val_idx,dtype=np.int32)
    test_idx=np.array(test_idx,dtype=np.int32)

    train_labels=train_labels.numpy().T
    val_labels=val_labels.numpy().T
    test_labels=test_labels.numpy().T

    train_labels=np.array(train_labels,dtype=np.int32)
    val_labels=np.array(val_labels,dtype=np.int32)
    test_labels=np.array(test_labels,dtype=np.int32)
    train_labels=train_labels.reshape(train_labels.shape[1])
    val_labels=val_labels.reshape(val_labels.shape[1])
    test_labels=test_labels.reshape(test_labels.shape[1])
    
    np.savez(root_folder+'/data/mooc/mooc_labels.npz',train_idx=train_idx,val_idx=val_idx,test_idx=test_idx,train_labels=train_labels,val_labels=val_labels,test_labels=test_labels,all_labels = labels.data)

    # print("data_ogb.edge_index", data_ogb.edge_index.shape, data_ogb.edge_index)
    data.edge_index = torch.tensor(data.edge_index).squeeze(0)
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row_ful,col_ful=data.edge_index
    row_ful=row_ful.numpy()
    col_ful=col_ful.numpy()
    edge_number = 0

    save_adj(row_ful, col_ful, N=data.num_nodes, dataset_name='mooc', savename='mooc_full', snap='init')
   
    data.edge_index, drop_edge_index, _ = dropout_adj(data.edge_index,train_idx, num_nodes= data.num_nodes)
    
    shuffle_index=torch.randperm(drop_edge_index.shape[0])

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    
    row_drop, col_drop = np.array(drop_edge_index)
    print('row_drop.shape:', row_drop.shape)
    f = open(root_folder+'/data/mooc/mooc_update_full.txt', 'w+')
    for k in range(row_drop.shape[0]):
        v_from = row_drop[k]
        v_to = col_drop[k]
        f.write('%d %d\n' % (v_from, v_to))
        f.write('%d %d\n' % (v_to, v_from))
    f.close()
    
    row,col=data.edge_index
    row=row.numpy()
    col=col.numpy()
    edge_number = 0
    with open(root_folder+'/data/mooc/mooc_init_adj' + '.txt', 'w') as f:
        if(self_loop):
            for i, j in zip(row, col):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
                edge_number+=1
        else:
            for i, j in zip(row, col):
                if(i != j):
                    f.write("%d %d\n" % (i, j))
                    f.write("%d %d\n" % (j, i))
                    edge_number+=1
    print("edge_number", edge_number) 


    save_adj(row, col, N=data.num_nodes, dataset_name='mooc', savename='mooc_init', snap='init')
    num_snap = 17
    snapshot = math.floor(row_drop.shape[0] / num_snap)
    print('num_snap: ', num_snap)

    for sn in range(num_snap):
        print(sn)
        row_sn = row_drop[ sn*snapshot : (sn+1)*snapshot ]
        col_sn = col_drop[ sn*snapshot : (sn+1)*snapshot ]
        if sn == 0:
            row_tmp=row
            col_tmp=col
        
        row_tmp=np.concatenate((row_tmp,row_sn))
        col_tmp=np.concatenate((col_tmp,col_sn))
        row_tmp=np.concatenate((row_tmp,col_sn))
        col_tmp=np.concatenate((col_tmp,row_sn))
        
        save_adj(row_tmp, col_tmp, N=data.num_nodes, dataset_name='mooc', savename='mooc_snap'+str(sn+1), snap=(sn+1))
        
        with open(root_folder+'/data/mooc/mooc_Edgeupdate_snap' + str(sn+1) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print('mooc -- save snapshots finish')

def wikipedia():
    # dataset=PygNodePropPredDataset(name='ogbn-products',root="../data/")
    # dataset_ogb=PygNodePropPredDataset(name='ogbn-arxiv',root ="/Resource/dataset/OGB")
    # data_ogb = dataset_ogb[0]


    data = dataset.wikipedia()
    # data.split_nodes(train_size=0.7, val_size=0.1,
    #              test_size=0.2, random_state=42)
    
    
    
    # train_idx, val_idx, test_idx = data.train_nodes, data.val_nodes, data.test_nodes
    # # all_idx = torch.cat([train_idx, val_idx, test_idx])

    train_idx = np.load(root_folder+'/dataset/wikipedia/train_wikipedia.npy')
    val_idx = np.load(root_folder+'/dataset/wikipedia/val_wikipedia.npy')
    test_idx = np.load(root_folder+'/dataset/wikipedia/test_wikipedia.npy')
    all_size = np.arange(data.num_nodes)
    # train_idx = all_size[train_idx]
    # val_idx = all_size[val_idx]
    # test_idx = all_size[test_idx]

    print("new_train_idx",train_idx.shape)
    print("new_train_idx",val_idx.shape)
    print("new_train_idx",test_idx.shape)
    # exit(0)
    
    # exit(0)
    #get labels
    print("save labels.....")
    data.y = data.y.unsqueeze(1)
    
    print("data.y", data.y.shape)

    labels=data.y
    train_labels=labels.data[train_idx]
    val_labels=labels.data[val_idx]
    test_labels=labels.data[test_idx]

    print("np.sum(train_labels)", torch.sum(train_labels))
    # exit(0)
    # train_idx=train_idx.numpy()
    # val_idx=val_idx.numpy()
    # test_idx=test_idx.numpy()
    train_idx=np.array(train_idx, dtype=np.int32)
    val_idx=np.array(val_idx,dtype=np.int32)
    test_idx=np.array(test_idx,dtype=np.int32)

    train_labels=train_labels.numpy().T
    val_labels=val_labels.numpy().T
    test_labels=test_labels.numpy().T

    train_labels=np.array(train_labels,dtype=np.int32)
    val_labels=np.array(val_labels,dtype=np.int32)
    test_labels=np.array(test_labels,dtype=np.int32)
    train_labels=train_labels.reshape(train_labels.shape[1])
    val_labels=val_labels.reshape(val_labels.shape[1])
    test_labels=test_labels.reshape(test_labels.shape[1])
    
    np.savez(root_folder+'/data/wikipedia/wikipedia_labels.npz',train_idx=train_idx,val_idx=val_idx,test_idx=test_idx,train_labels=train_labels,val_labels=val_labels,test_labels=test_labels,all_labels = labels.data)

    # print("data_ogb.edge_index", data_ogb.edge_index.shape, data_ogb.edge_index)
    data.edge_index = torch.tensor(data.edge_index).squeeze(0)
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row_ful,col_ful=data.edge_index
    row_ful=row_ful.numpy()
    col_ful=col_ful.numpy()
    edge_number = 0

    save_adj(row_ful, col_ful, N=data.num_nodes, dataset_name='wikipedia', savename='wikipedia_full', snap='init')
   
    data.edge_index, drop_edge_index, _ = dropout_adj(data.edge_index,train_idx, num_nodes= data.num_nodes)
    
    shuffle_index=torch.randperm(drop_edge_index.shape[0])

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    
    row_drop, col_drop = np.array(drop_edge_index)
    print('row_drop.shape:', row_drop.shape)
    f = open(root_folder+'/data/wikipedia/wikipedia_update_full.txt', 'w+')
    for k in range(row_drop.shape[0]):
        v_from = row_drop[k]
        v_to = col_drop[k]
        f.write('%d %d\n' % (v_from, v_to))
        f.write('%d %d\n' % (v_to, v_from))
    f.close()
    
    row,col=data.edge_index
    row=row.numpy()
    col=col.numpy()
    edge_number = 0
    with open(root_folder+'/data/wikipedia/wikipedia_init_adj' + '.txt', 'w') as f:
        if(self_loop):
            for i, j in zip(row, col):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
                edge_number+=1
        else:
            for i, j in zip(row, col):
                if(i != j):
                    f.write("%d %d\n" % (i, j))
                    f.write("%d %d\n" % (j, i))
                    edge_number+=1
    print("edge_number", edge_number) 


    save_adj(row, col, N=data.num_nodes, dataset_name='wikipedia', savename='wikipedia_init', snap='init')
    num_snap = 17
    snapshot = math.floor(row_drop.shape[0] / num_snap)
    print('num_snap: ', num_snap)

    for sn in range(num_snap):
        print(sn)
        row_sn = row_drop[ sn*snapshot : (sn+1)*snapshot ]
        col_sn = col_drop[ sn*snapshot : (sn+1)*snapshot ]
        if sn == 0:
            row_tmp=row
            col_tmp=col
        
        row_tmp=np.concatenate((row_tmp,row_sn))
        col_tmp=np.concatenate((col_tmp,col_sn))
        row_tmp=np.concatenate((row_tmp,col_sn))
        col_tmp=np.concatenate((col_tmp,row_sn))
        
        save_adj(row_tmp, col_tmp, N=data.num_nodes, dataset_name='wikipedia', savename='wikipedia_snap'+str(sn+1), snap=(sn+1))
        
        with open(root_folder+'/data/wikipedia/wikipedia_Edgeupdate_snap' + str(sn+1) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print('wikipedia -- save snapshots finish')

def reddit():
    # dataset=PygNodePropPredDataset(name='ogbn-products',root="../data/")
    # dataset_ogb=PygNodePropPredDataset(name='ogbn-arxiv',root ="/Resource/dataset/OGB")
    # data_ogb = dataset_ogb[0]


    data = dataset.reddit()
    # data.split_nodes(train_size=0.7, val_size=0.1,
    #              test_size=0.2, random_state=42)
    
    
    
    # train_idx, val_idx, test_idx = data.train_nodes, data.val_nodes, data.test_nodes
    # # all_idx = torch.cat([train_idx, val_idx, test_idx])

    train_idx = np.load(root_folder+'/dataset/reddit/train_reddit.npy')
    val_idx = np.load(root_folder+'/dataset/reddit/val_reddit.npy')
    test_idx = np.load(root_folder+'/dataset/reddit/test_reddit.npy')
    all_size = np.arange(data.num_nodes)
    # train_idx = all_size[train_idx]
    # val_idx = all_size[val_idx]
    # test_idx = all_size[test_idx]

    print("new_train_idx",train_idx.shape)
    print("new_train_idx",val_idx.shape)
    print("new_train_idx",test_idx.shape)
    # exit(0)
    
    # exit(0)
    #get labels
    print("save labels.....")
    data.y = data.y.unsqueeze(1)
    
    print("data.y", data.y.shape)

    labels=data.y
    train_labels=labels.data[train_idx]
    val_labels=labels.data[val_idx]
    test_labels=labels.data[test_idx]
    
    print("np.sum(train_labels)", torch.sum(train_labels))

    # train_idx=train_idx.numpy()
    # val_idx=val_idx.numpy()
    # test_idx=test_idx.numpy()
    train_idx=np.array(train_idx, dtype=np.int32)
    val_idx=np.array(val_idx,dtype=np.int32)
    test_idx=np.array(test_idx,dtype=np.int32)

    train_labels=train_labels.numpy().T
    val_labels=val_labels.numpy().T
    test_labels=test_labels.numpy().T

    train_labels=np.array(train_labels,dtype=np.int32)
    val_labels=np.array(val_labels,dtype=np.int32)
    test_labels=np.array(test_labels,dtype=np.int32)
    train_labels=train_labels.reshape(train_labels.shape[1])
    val_labels=val_labels.reshape(val_labels.shape[1])
    test_labels=test_labels.reshape(test_labels.shape[1])
    
    np.savez(root_folder+'/data/reddit/reddit_labels.npz',train_idx=train_idx,val_idx=val_idx,test_idx=test_idx,train_labels=train_labels,val_labels=val_labels,test_labels=test_labels,all_labels = labels.data)

    # print("data_ogb.edge_index", data_ogb.edge_index.shape, data_ogb.edge_index)
    data.edge_index = torch.tensor(data.edge_index).squeeze(0)
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row_ful,col_ful=data.edge_index
    row_ful=row_ful.numpy()
    col_ful=col_ful.numpy()
    edge_number = 0

    save_adj(row_ful, col_ful, N=data.num_nodes, dataset_name='reddit', savename='reddit_full', snap='init')
   
    data.edge_index, drop_edge_index, _ = dropout_adj(data.edge_index,train_idx, num_nodes= data.num_nodes)
    
    shuffle_index=torch.randperm(drop_edge_index.shape[0])

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    
    row_drop, col_drop = np.array(drop_edge_index)
    print('row_drop.shape:', row_drop.shape)
    f = open(root_folder+'/data/reddit/reddit_update_full.txt', 'w+')
    for k in range(row_drop.shape[0]):
        v_from = row_drop[k]
        v_to = col_drop[k]
        f.write('%d %d\n' % (v_from, v_to))
        f.write('%d %d\n' % (v_to, v_from))
    f.close()
    
    row,col=data.edge_index
    row=row.numpy()
    col=col.numpy()
    edge_number = 0
    with open(root_folder+'/data/reddit/reddit_init_adj' + '.txt', 'w') as f:
        if(self_loop):
            for i, j in zip(row, col):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
                edge_number+=1
        else:
            for i, j in zip(row, col):
                if(i != j):
                    f.write("%d %d\n" % (i, j))
                    f.write("%d %d\n" % (j, i))
                    edge_number+=1
    print("edge_number", edge_number) 


    save_adj(row, col, N=data.num_nodes, dataset_name='reddit', savename='reddit_init', snap='init')
    num_snap = 17
    snapshot = math.floor(row_drop.shape[0] / num_snap)
    print('num_snap: ', num_snap)

    for sn in range(num_snap):
        print(sn)
        row_sn = row_drop[ sn*snapshot : (sn+1)*snapshot ]
        col_sn = col_drop[ sn*snapshot : (sn+1)*snapshot ]
        if sn == 0:
            row_tmp=row
            col_tmp=col
        
        row_tmp=np.concatenate((row_tmp,row_sn))
        col_tmp=np.concatenate((col_tmp,col_sn))
        row_tmp=np.concatenate((row_tmp,col_sn))
        col_tmp=np.concatenate((col_tmp,row_sn))
        
        save_adj(row_tmp, col_tmp, N=data.num_nodes, dataset_name='reddit', savename='reddit_snap'+str(sn+1), snap=(sn+1))
        
        with open(root_folder+'/data/reddit/reddit_Edgeupdate_snap' + str(sn+1) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print('reddit -- save snapshots finish')

def mag():
    print('start processing data: ')
    # dataset=PygNodePropPredDataset(name='ogbn-mag')
    dataset=PygNodePropPredDataset(name='ogbn-mag',root ="/Resource/dataset/OGB")
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx['train']['paper'], split_idx['valid']['paper'], split_idx['test']['paper']
    
    # print("data:",data)
    feat=data.x_dict["paper"].numpy()
    feat=np.array(feat,dtype=np.float64)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)
    print("feat.size()", feat.shape)
    np.save(root_folder+'/data/mag/mag_feat.npy',feat)
    
    print("train_idx, ", train_idx.shape)
    #get labels
    labels=data.y_dict["paper"]
    print(labels)
    train_labels=labels.data[train_idx]
    val_labels=labels.data[val_idx]
    test_labels=labels.data[test_idx]

    train_idx=train_idx.numpy()
    val_idx=val_idx.numpy()
    test_idx=test_idx.numpy()
    train_idx=np.array(train_idx, dtype=np.int32)
    val_idx=np.array(val_idx,dtype=np.int32)
    test_idx=np.array(test_idx,dtype=np.int32)

    train_labels=train_labels.numpy().T
    val_labels=val_labels.numpy().T
    test_labels=test_labels.numpy().T

    train_labels=np.array(train_labels,dtype=np.int32)
    val_labels=np.array(val_labels,dtype=np.int32)
    test_labels=np.array(test_labels,dtype=np.int32)
    train_labels=train_labels.reshape(train_labels.shape[1])
    val_labels=val_labels.reshape(val_labels.shape[1])
    test_labels=test_labels.reshape(test_labels.shape[1])
    np.savez(root_folder+'/data/mag/mag_labels.npz',train_idx=train_idx,val_idx=val_idx,test_idx=test_idx,train_labels=train_labels,val_labels=val_labels,test_labels=test_labels)
    
    edge_index=data.edge_index_dict[("paper", "cites", "paper")]
    print("before undirect edge_index.shape:", edge_index.shape)
    data.edge_index = to_undirected(edge_index, data.num_nodes_dict["paper"])
    print("after undirect edge_index.shape:", edge_index.shape)


    row_ful,col_ful=data.edge_index
    row_ful=row_ful.numpy()
    col_ful=col_ful.numpy()
    edge_number = 0
    with open(root_folder+'/data/mag/mag_full_adj' + '.txt', 'w') as f:
        # if(self_loop):
        for i, j in zip(row_ful, col_ful):
            f.write("%d %d\n" % (i, j))
            f.write("%d %d\n" % (j, i))
            edge_number+=1
        # else:
        #     for i, j in zip(row_ful, col_ful):
        #         if(i != j):
        #             f.write("%d %d\n" % (i, j))
        #             f.write("%d %d\n" % (j, i))
        #             edge_number+=1
    print("full edge_number", edge_number)
    save_adj(row_ful, col_ful, N=data.num_nodes, dataset_name='mag', savename='mag_full', snap='init')

    
    edge_index, drop_edge_index, _ = dropout_adj(edge_index,train_idx, num_nodes= data.num_nodes_dict["paper"])
    
    edge_index = to_undirected(edge_index, data.num_nodes_dict["paper"])
    
    row_drop, col_drop = np.array(drop_edge_index)
    print('row_drop.shape:', row_drop.shape)
    f = open(root_folder+'/data/mag/ogbn-mag_update_full.txt', 'w+')
    for k in range(row_drop.shape[0]):
        v_from = row_drop[k]
        v_to = col_drop[k]
        f.write('%d %d\n' % (v_from, v_to))
        f.write('%d %d\n' % (v_to, v_from))
    f.close()
    
    row,col=edge_index
    row=row.numpy()
    col=col.numpy()

    edge_number = 0
    with open(root_folder+'/data/mag/mag_init_adj' + '.txt', 'w') as f:
        if(self_loop):
            for i, j in zip(row, col):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
                edge_number+=1
        else:
            for i, j in zip(row, col):
                if(i != j):
                    f.write("%d %d\n" % (i, j))
                    f.write("%d %d\n" % (j, i))
                    edge_number+=1
    print("edge_number", edge_number) 
    
    save_adj(row, col, N=data.num_nodes_dict["paper"], dataset_name='mag', savename='mag_init', snap='init')
    num_snap = 16
    snapshot = math.floor(row_drop.shape[0] / num_snap)
    print('num_snap: ', num_snap)
    # #Shuffle the removed edges and show a more flexible setting
    # # Generate shuffled indices
    # indices = np.random.permutation(len(row_drop))
    # # Shuffle both arrays using the same indices
    # row_drop = row_drop[indices]
    # col_drop = col_drop[indices]

    print('num_snap: ', row_drop)

    for sn in range(num_snap):
        print(sn)
        row_sn = row_drop[ sn*snapshot : (sn+1)*snapshot ]
        col_sn = col_drop[ sn*snapshot : (sn+1)*snapshot ]
        if sn == 0:
            row_tmp=row
            col_tmp=col
        
        row_tmp=np.concatenate((row_tmp,row_sn))
        col_tmp=np.concatenate((col_tmp,col_sn))
        row_tmp=np.concatenate((row_tmp,col_sn))
        col_tmp=np.concatenate((col_tmp,row_sn))
        
        save_adj(row_tmp, col_tmp, N=data.num_nodes_dict["paper"], dataset_name='mag', savename='mag_snap'+str(sn+1), snap=(sn+1))
        
        with open(root_folder+'/data/mag/mag_Edgeupdate_snap' + str(sn+1) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print('mag -- save snapshots finish')

def save_adj(row, col, N, dataset_name, savename, snap, full=False):
    adj=sp.csr_matrix((np.ones(row.shape[0]),(row,col)),shape=(N,N))
    if(snap == "init"):
        if(self_loop):
            adj=adj+sp.eye(adj.shape[0])
    print('snap:',snap,', edge:',adj.nnz)
    save_path=root_folder+'/data/'+ dataset_name +'/'

    EL=adj.indices
    PL=adj.indptr

    del adj
    gc.collect()

    EL=np.array(EL,dtype=np.uint32)
    PL=np.array(PL,dtype=np.uint32)
    EL_re=[]

    for i in range(1,PL.shape[0]):
        EL_re+=sorted(EL[PL[i-1]:PL[i]],key=lambda x:PL[x+1]-PL[x])
    EL_re=np.asarray(EL_re,dtype=np.uint32)

    #save graph
    f1=open(save_path+savename+'_adj_el.txt','wb')
    for i in EL_re:
        m=struct.pack('I',i)
        f1.write(m)
                
    f1.close()

    f2=open(save_path+savename+'_adj_pl.txt','wb')
    for i in PL:
        m=struct.pack('I',i)
        f2.write(m)
    f2.close()
    del EL
    del PL
    del EL_re
    gc.collect()

if __name__ == "__main__":
    # papers100M_edge()
    # products()
    # patent()
    # arxiv()
    # tmall()
    mag()
    # mooc()
    # wikipedia()
    # reddit()