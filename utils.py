import torch
import gc
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from propagation import InstantGNN
import time
import copy
import pdb
import sklearn.preprocessing
def add_log(pth, log):
  time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
  with open(pth, "a+") as f:
    f.write(time_now + ": " + str(log) + "\n")

def load_aminer_init(datastr, rmax, alpha):
    if datastr == "1984_author_dense":
        m = 3787605; n = 1252095
    elif datastr == "2013_author_dense":
        m = 9237799; n = 1252095

    print("Load %s!" % datastr)
    labels = np.load("./data/aminer/"+ datastr +"_labels.npy")
    
    py_alg = InstantGNN()

    features = np.load('./data/aminer/aminer_dense_feat.npy')
    memory_dataset = py_alg.initial_operation('./data/aminer/',datastr, m, n, rmax, alpha, features)
    split = np.load('./data/aminer/aminer_dense_idx_split.npz')
    train_idx, val_idx, test_idx = split['train'], split['valid'], split['test']
    
    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    
    train_labels = torch.LongTensor(labels[train_idx])
    val_labels = torch.LongTensor(labels[val_idx])
    test_labels = torch.LongTensor(labels[test_idx])

    train_labels = train_labels.reshape(train_labels.size(0), 1)
    val_labels = val_labels.reshape(val_labels.size(0), 1)
    test_labels = test_labels.reshape(test_labels.size(0), 1)
    
    return features, train_labels, val_labels, test_labels, train_idx, val_idx, test_idx, memory_dataset, py_alg

def load_ogb_init(datastr, alpha, rmax, epsilon, algorithm):
    if(datastr=="papers100M"):
        m=3259203018; n=111059956 ##init graph
    elif(datastr=="arxiv"):
        # m=427696; n=169343
        m=597039; n=169343 ##with self_loop
        # m=2315598; n=169343 ## full edge
    elif(datastr=="products"):
        # m=69634445; n=2449029
        m=67185416; n=2449029
    print("Load %s!" % datastr)
    
    py_alg = InstantGNN()


    features = np.load('./data/'+datastr+'/'+datastr+'_feat.npy')
    print("load the original feat....")


    # features = np.load('./data/'+datastr+'/'+datastr+'_feat_norm.npy')
    # print("load the norm feat....")


    features=np.array(features,dtype=np.float64)
    perm = torch.randperm(features.shape[0])
    features_n = features[perm]
    
    print("features.size()",np.size(features,0),"  ", np.size(features,1))
    
    print("features[357[37]",features[357][37])
    memory_dataset = py_alg.initial_operation('./data/'+datastr+'/', datastr+'_init', m, n, rmax, alpha, epsilon,features, algorithm)
    memory_dataset_n = py_alg.initial_operation('./data/'+datastr+'/', datastr+'_init', m, n, rmax, alpha, epsilon,features_n, algorithm)
    # scaler = sklearn.preprocessing.StandardScaler()
    # scaler.fit(features)
    # features = scaler.transform(features)
  

    data = np.load('./data/'+datastr+'/'+datastr+'_labels.npz')
    train_idx = torch.LongTensor(data['train_idx'])
    val_idx = torch.LongTensor(data['val_idx'])
    test_idx =torch.LongTensor(data['test_idx'])
    
    print("train_idx", train_idx)
    train_labels = torch.LongTensor(data['train_labels'])
    val_labels = torch.LongTensor(data['val_labels'])
    test_labels = torch.LongTensor(data['test_labels'])
    train_labels=train_labels.reshape(train_labels.size(0),1)
    val_labels=val_labels.reshape(val_labels.size(0),1)
    test_labels=test_labels.reshape(test_labels.size(0),1)
    
    return n,m,features,features_n,train_labels,val_labels,test_labels,train_idx,val_idx,test_idx,memory_dataset, py_alg

def load_sbm_init(datastr, rmax, alpha):
    if datastr == "SBM-50000-50-20+1":
        m=1412466; n=50000
    elif datastr == "SBM-500000-50-20+1":
        m=14141662; n=500000
    elif datastr == "SBM-10000000-100-20+1":
        m=282938572;n=10000000
    elif datastr == "SBM-1000000-50-20+1":
        m=28293138;n=1000000

    print("Load %s!" % datastr)

    labels = np.loadtxt('./data/'+datastr+'/'+datastr+'_label.txt')
    
    py_alg = InstantGNN()
    
    if datastr == "SBM-1000000-50-20+1" or datastr== "SBM-500000-50-20+1":
        encode_len = 256
    else:
        encode_len = 1024
    
    split = np.load('./data/'+datastr+'/'+datastr+'_idx_split.npz')
    train_idx, val_idx, test_idx = split['train'], split['valid'], split['test']
    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
        
    features = np.load('./data/'+datastr+'/'+datastr+'_encode_'+str(encode_len)+'_feat.npy')
    memory_dataset = py_alg.initial_operation('./data/'+datastr+'/adjs/', datastr+'_init', m, n, rmax, alpha, features)
    
    train_labels = torch.LongTensor(labels[train_idx])
    val_labels = torch.LongTensor(labels[val_idx])
    test_labels = torch.LongTensor(labels[test_idx])

    train_labels = train_labels.reshape(train_labels.size(0), 1)
    val_labels = val_labels.reshape(val_labels.size(0), 1)
    test_labels = test_labels.reshape(test_labels.size(0), 1)
    
    return features, train_labels, val_labels, test_labels, train_idx, val_idx, test_idx, memory_dataset, py_alg

def muticlass_f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    macro = f1_score(labels, preds, average='macro')
    return macro

def com_accuracy(y_pred, y):
    pred = y_pred.data.max(1)[1]
    pred = pred.reshape(pred.size(0),1)
    correct = pred.eq(y.data).cpu().sum()
    accuracy = correct.to(dtype=torch.long) * 100. / len(y)
    return accuracy

# Build the dataset for the positive and negtive data, here x.szie()!= x_n.size()
# class ExtendDataset(Dataset):
#     def __init__(self,x,x_n):
#         self.x=x
#         self.x_n = x_n
#         self.ratio = self.x.size(0)/self.x_n.size(0)
#     def __len__(self):
#         return self.x_n.size(0)

#     def __getitem__(self,idx):
#         return self.x[(idx*self.ratio)],self.x_n[idx]

class SimpleDataset(Dataset):
    def __init__(self,x,x_n,y):
        self.x=x
        self.x_n = x_n
        self.y=y
        # assert self.x_n.size(0)==self.y.size(0)
        # assert self.x.size(0)==self.x_n.size(0)
    def __len__(self):
        return self.x.size(0)

    def __getitem__(self,idx):
        return self.x[idx],self.x_n[idx],self.y[idx]

