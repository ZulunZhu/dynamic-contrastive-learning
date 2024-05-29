import time
import uuid
import random
import argparse
import gc
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ogb.nodeproppred import Evaluator
from utils import SimpleDataset
from model import ClassMLP,PGL,Classifier,GGD, GGD_Encoder
from utils import *
from glob import glob
from tqdm import tqdm
from sklearn import preprocessing as sk_prep
from sklearn import metrics
import logging
import copy
import os
import sys
from sys import getsizeof
import update_grad
import pynvml
import resource
from sklearn.metrics import average_precision_score, roc_auc_score
from openTSNE import TSNE
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from varname import nameof
def show_gpu():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 0表示显卡标号
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print("GPU overall::",meminfo.total/1024**3, "GB") #总的显存大小
    print("GPU allocated::",meminfo.used/1024**3, "GB")  #已用显存大小
    print("GPU left::",meminfo.free/1024**3, "GB")  #剩余显存大小



def update_results_csv(result_path, model_name, dataset_name, new_result):
    # Load the existing CSV file into a DataFrame or create a new one if it doesn't exist
    try:
        df = pd.read_csv(result_path, index_col=0)
    except FileNotFoundError:
        # If the file does not exist, create an empty DataFrame with model_name as the index
        df = pd.DataFrame(columns=[dataset_name])
        df.index.name = 'Model'

    # Check if the dataset_name column exists, if not, add it
    if dataset_name not in df.columns:
        df[dataset_name] = pd.NA  # Initialize the column with NA values

    # Ensure the model_name row exists
    if model_name not in df.index:
        # If the model doesn't exist, append a new row with NA values
        new_row = pd.DataFrame(index=[model_name], columns=df.columns)
        df = pd.concat([df, new_row])

    # Update the specific entry with the new result
    df.at[model_name, dataset_name] = new_result

    # Save the updated DataFrame back to CSV
    df.to_csv(result_path)

    print(f"Updated CSV file at {result_path} with new results for model '{model_name}' and dataset '{dataset_name}'.")

def tsne_plt(embeddings, labels, save_path=None, title='Title'):
    print('Drawing t-SNE plot ...')
    tsne = TSNE(n_components=3, perplexity=30, metric="euclidean", n_jobs=8, random_state=42, verbose=False)
    embeddings = embeddings.cpu().numpy()
    c = labels.cpu().numpy()

    emb = tsne.fit(embeddings)  # Training

    plt.figure(figsize=(10, 8))
    plt.scatter(emb[:, 0], emb[:, 1], c=c, marker='o')
    plt.colorbar()
    plt.grid(True)
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def main():
    parser = argparse.ArgumentParser()
    
    
    # Dataset and Algorithom
    parser.add_argument('--seed', type=int, default=20159, help='random seed..')
    parser.add_argument('--alg', default='instant', help='push algorithm')
    parser.add_argument('--cl_alg', default='PGL', help='contrastive learning algorithm')
    parser.add_argument('--dataset', default='papers100M', help='dateset.')
    # Algorithm parameters
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha.')
    parser.add_argument('--rmax', type=float, default=1e-7, help='threshold.')
    parser.add_argument('--rbmax', type=float, default=1, help='reverse push threshold.')
    parser.add_argument('--delta', type=float, default=1, help='positive sample threshold.')

    parser.add_argument('--epsilon', type=float, default=8, help='epsilon.')
    parser.add_argument("--n-ggd-epochs", type=int, default=1,
                        help="number of training epochs")
    parser.add_argument("--n-classifier-epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--drop_feat", type=float, default=0.1,
                        help="feature dropout rate")
    parser.add_argument('--use_gcl', default="yes", help='bias.')
    # Learining parameters
    
    parser.add_argument("--classifier-lr", type=float, default=0.05, help="classifier learning rate")
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay.')
    parser.add_argument('--layer', type=int, default=4, help='number of layers.')
    parser.add_argument('--hidden', type=int, default=2048, help='hidden dimensions.')
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate.')
    parser.add_argument('--bias', default='none', help='bias.')
    parser.add_argument("--proj_layers", type=int, default=1, help="number of project linear layers")
    parser.add_argument('--epochs', type=int, default= 10, help='number of epochs.')
    parser.add_argument('--batch', type=int, default=2048, help='batch size.')
    parser.add_argument("--patience", type=int, default=100, help="early stop patience condition")
    parser.add_argument('--dev', type=int, default=0, help='device id.')
    parser.add_argument('--skip_sn0', type=int, default=1, help='decide whether to skip snapshot 0.')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print("--------------------------")
    print(args)

    free_gpu_id = int(get_free_gpu())
    torch.cuda.set_device(free_gpu_id)
    # torch.cuda.set_device(args.dev)
    checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'

    
    n,m,features,features_n,train_labels,val_labels,test_labels,labels,train_idx,val_idx,test_idx,memory_dataset, py_alg = load_ogb_init(args.dataset, args.alpha,args.rmax, args.rbmax,args.delta,args.epsilon, args.alg) ##
    print("features::",features[1:20])
    print("train_labels",torch.sum(train_labels))
    features_p = features
    print('------------------ Initial -------------------')
    print("train_idx:",train_idx.size())
    print("val_idx:",val_idx.size())
    print("test_idx:",test_idx.size())
    macros = []
    micros = [] 
    pretrain_times = []
    change_node_list = np.zeros([n]) 
    if not args.skip_sn0:
        macro_init, micro_init, pretrain_time_init = prepare_to_train(0,features,features_p,features_n, m, train_idx, val_idx, test_idx, train_labels, val_labels, test_labels, labels,args, checkpt_file, change_node_list)
        macros.append(macro_init)
        micros.append(micro_init)
        pretrain_times.append(pretrain_time_init)
    arxiv_table_path = Path('../mag_accuracy_table.csv')

    print('------------------ update -------------------')
    snapList = [f for f in glob('../data/'+args.dataset+'/*Edgeupdate_32snap*.txt')]
    print('number of snapshots: ', len(snapList))
    # print("features[2][3]::",features[2][3])
    for i in range(len(snapList)):
        change_node_list = np.zeros([n]) 
        features_copy = copy.deepcopy(features)
        # print("change_node_list",np.sum(change_node_list))
        py_alg.snapshot_operation('../data/'+args.dataset+'/'+args.dataset+'_Edgeupdate_32snap'+str(i+1)+'.txt', args.rmax, args.alpha, features, args.alg)
        print("number of changed node", np.sum(change_node_list))
        # print("features_ori:",features)
        change_node_list = change_node_list.astype(bool)
        print("change_node_list:",change_node_list.shape)
        features_p = features_copy[~change_node_list]
        features_n = features_copy[change_node_list]
        print("feature_p.size:",features_p.shape)
        print("feature_n.size:",features_n.shape)

        # # if(i == 1):
        macro, micro, pretrain_time = prepare_to_train(i+1,features, features_p, features_n, m,train_idx, val_idx, test_idx, train_labels, val_labels, test_labels, labels, args, checkpt_file, change_node_list)
        macros.append(macro)
        micros.append(micro)
        pretrain_times.append(pretrain_time)
        # # update_results_csv(arxiv_table_path, "negative samples", str(args.delta), features_n.shape[0])
        # update_results_csv(arxiv_table_path, "accuracy", i, micro)
        

        # with open('./log/sensitivity.txt', 'a') as f:
        #     print('Dataset:'+args.dataset+f"metric_micro:{100*np.mean( micros):.2f}%  "+f" hidden: {args.hidden:.1f}"+f" epochs: {args.n_ggd_epochs:.1f}",file=f)
        # exit(0)
        del features_copy
        gc.collect()
    print("Mean Macro: ", np.mean( macros), " Mean Micro: ", np.mean( micros), "Mean training time: ", np.mean(pretrain_times))
    # with open('./log/sensitivity.txt', 'a') as f:
    #     print('Dataset:'+args.dataset+f"metric_micro:{100*np.mean( micros):.2f}%  "+f" hidden: {args.hidden:.1f}"+f" epochs: {args.n_ggd_epochs:.1f}",file=f)

def train_incremental(model, device, train_loader, optimizer, mem_loader):
    model.train()

    time_epoch=0
    loss_list=[]
    loss_fun = nn.BCEWithLogitsLoss()
    for step, (x, y) in enumerate(train_loader):
        t_st=time.time()
        task_grads = {}
        for step_mem, (x_mem, y_mem) in enumerate(mem_loader):
            x_mem, y_mem = x_mem.cuda(), y_mem.cuda()
            optimizer.zero_grad()
            out_mem = model(x_mem)
            loss_mem = F.nll_loss(out_mem, y_mem.squeeze(1))
            # print("out", out.sigmoid())
            # print("y.squeeze(1)", y.to(torch.float))
            # if(torch.sum(y)>0):
            #     print("torch.sum(y)>0!!")
            # loss = loss_fun(out.sigmoid(), y.to(torch.float))
            loss_mem.backward()
            gradients = {}
            for name, parameter in model.named_parameters():
                gradients[name] = parameter.grad.clone()
            task_grads[step_mem] = update_grad.grad_to_vector(gradients)
        ref_grad_vec = torch.stack(list(task_grads.values()))
        ref_grad_vec = torch.sum(ref_grad_vec, dim=0)/ref_grad_vec.shape[0]

        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out, y.squeeze(1))
        # print("out", out.sigmoid())
        # print("y.squeeze(1)", y.to(torch.float))
        # if(torch.sum(y)>0):
        #     print("torch.sum(y)>0!!")
        # loss = loss_fun(out.sigmoid(), y.to(torch.float))
        loss.backward()
        # for name, parameter in model.named_parameters():
        #     print(parameter.grad)
        # Example: save gradients as a torch file
        gradients = {}
        for name, parameter in model.named_parameters():
            gradients[name] = parameter.grad.clone()  # Use `.clone()` to save a copy of the gradient tensor
        # torch.save(gradients, grad_checkpt_file)
        # loaded_gradients = torch.load(grad_checkpt_file)

    #    for param in classifier.parameters():
    #     print(param)
    #    print("loaded_gradients", loaded_gradients)
        # for n, p in gradients.items():
        #     print("loaded_gradients", p)
        current_grad_vec = update_grad.grad_to_vector(gradients)
        # print("current_grad_vec", current_grad_vec)
        # print("ref_grad_vec", ref_grad_vec)

        assert current_grad_vec.shape == ref_grad_vec.shape
        dotp = current_grad_vec * ref_grad_vec
        dotp = dotp.sum()
        if (dotp < 0).sum() != 0:
            # new_grad = update_grad.get_grad(current_grad_vec, ref_grad_vec)
            new_grad = update_grad.get_grad(current_grad_vec,ref_grad_vec)
            # copy gradients back
            # print("current_grad_vec", current_grad_vec)
            # print("new_grad", new_grad)
            update_grad.vector_to_grad(model,new_grad)
            # for name, parameter in model.named_parameters():
            #     print(parameter.grad)
            # exit(0)
        
        optimizer.step()
        time_epoch+=(time.time()-t_st)
        loss_list.append(loss.item())
        
    return np.mean(loss_list), time_epoch

def train(model, device, train_loader, optimizer):
    model.train()

    time_epoch=0
    loss_list=[]
    loss_fun = nn.BCEWithLogitsLoss()
    for step, (x, y) in enumerate(train_loader):
        t_st=time.time()
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out, y.squeeze(1))
        # print("out", out.sigmoid())
        # print("y.squeeze(1)", y.to(torch.float))
        # if(torch.sum(y)>0):
        #     print("torch.sum(y)>0!!")
        # loss = loss_fun(out.sigmoid(), y.to(torch.float))
        loss.backward()
        optimizer.step()
        time_epoch+=(time.time()-t_st)
        loss_list.append(loss.item())
        
    return np.mean(loss_list), time_epoch


@torch.no_grad()
def validate(model, device, loader, evaluator):
    model.eval()
    y_pred, y_true = [], []
    for step,(x, y) in enumerate(loader):
        x = x.cuda()
        out = model(x)
        y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
        y_true.append(y)
    return evaluator.eval({
        "y_true": torch.cat(y_true, dim=0),
        "y_pred": torch.cat(y_pred, dim=0),
    })['acc']


@torch.no_grad()
def test(model, device, loader, evaluator,checkpt_file):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    y_pred, y_true = [], []
    for step,(x, y) in enumerate(loader):
        x = x.cuda()
        out = model(x)
        y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
        y_true.append(y)
    metric_macro = metrics.f1_score(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0), average='macro')
    metric_micro = metrics.f1_score(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0), average='micro')
    # For mooc and reddit datasets
    # roc = roc_auc_score(torch.cat(y_true, dim=0),torch.cat(y_pred, dim=0))
    roc = 0
    return evaluator.eval({
        "y_true": torch.cat(y_true, dim=0),
        "y_pred": torch.cat(y_pred, dim=0),
    })['acc'], metric_macro, metric_micro, roc

def aug_feature_dropout(input_feat, drop_percent=0.2):
    # aug_input_feat = copy.deepcopy((input_feat.squeeze(0)))
    aug_input_feat = copy.deepcopy(input_feat)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_percent)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0

    return aug_input_feat

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def prepare_to_train(snapshot, features_ori, features_p, features_n,m, train_idx, val_idx, test_idx, train_labels, val_labels, test_labels, ori_all_labels,args, checkpt_file, change_node_list):
    features_ori = torch.FloatTensor(features_ori)
    features_n = torch.FloatTensor(features_n)
    features_p = torch.FloatTensor(features_p)
    n = features_ori.size()[0]
    feature_dim = features_ori.size(-1)
    print("Original feature size: ", features_ori.size(0))
    print("train_idx:", train_idx.size())
    print("n=", n, " m=", m)
    all_labels = torch.zeros(features_ori.size(0),dtype=torch.int64)
    if(snapshot>0):    
       features = torch.cat((features_ori,features_p),dim=0)
    #    features_n = torch.cat((features_n,features_n[:features.size(0)-features_ori.size(0), :]),dim=0)## Not finished yet
    #    print("Added positive feature size: ", features.size(0)-features_ori.size(0))
    #    assert features.size(0)==features_n.size(0)
       
    else:
        features = features_ori




    label_dim = int(max(train_labels.max(),val_labels.max(),test_labels.max()))+1
    labels = torch.cat((train_labels, val_labels,test_labels)).squeeze(1).cuda()
    ori_all_labels = ori_all_labels.squeeze(1).cuda()
    print("labels:",labels.size())
    
    
    # train_dataset = SimpleDataset(features_train,train_labels)
    # valid_dataset = SimpleDataset(features_val,val_labels)
    # test_dataset = SimpleDataset(features_test, test_labels)
    print("features.size(0):", features.size(0))
    print("features_n.size(0):", features_n.size(0))
    fake_labels_1 = torch.ones(features.size(0))
    fake_labels_0 = torch.zeros(features_n.size(0))
    fake_labels = torch.cat((fake_labels_1,fake_labels_0),dim=0)

    print("fake_labels.size(0):", fake_labels.size(0))
    print("labels.size(0):", labels.size(0))

    if(args.dataset=="tmall"):
        print("train_idx.size()", train_idx.size())
        print("train_labels.size()",train_labels.size())

        all_labels.scatter_(0,train_idx,train_labels.squeeze(1))
        all_labels.scatter_(0,val_idx,val_labels.squeeze(1))
        all_labels.scatter_(0,test_idx,test_labels.squeeze(1))

        print("all_labels[train_idx]", all_labels[train_idx])
        all_dataset_ori = ExtendDataset(features_ori,all_labels)
        all_loader_ori = DataLoader(all_dataset_ori, batch_size=args.batch,shuffle=False)
    elif(args.dataset in ["mooc", "wikipedia", "reddit"]):
        print("train_idx.size()", train_idx.size())
        # all_size = torch.cat((train_idx,val_idx,test_idx),dim=0)
        # features_ori =  features_ori[all_size]
        # print("features_ori[train_idx]", features_ori[train_idx].shape)
        train_idx = train_idx>0
        val_idx = val_idx>0
        test_idx = test_idx>0
        # print("features_ori[train_idx]", features_ori[train_idx].shape)
        # label_dim = int(max(train_labels.max(),val_labels.max(),test_labels.max()))
        all_dataset_ori = ExtendDataset(features_ori,ori_all_labels)
        all_loader_ori = DataLoader(all_dataset_ori, batch_size=args.batch,shuffle=False)
    else:
        all_dataset_ori = ExtendDataset(features_ori,labels)
        all_loader_ori = DataLoader(all_dataset_ori, batch_size=args.batch,shuffle=False)
    
    print("features_ori.shape:", features_ori.shape)
    print("ori_all_labels.shape",ori_all_labels.shape)
    # all_dataset = SimpleDataset(features,features_n,fake_labels)
    # all_loader = DataLoader(all_dataset, batch_size=args.batch,shuffle=False)
    
    #Cat all the feature
    features = torch.cat((features,features_n),dim=0)
    all_dataset = ExtendDataset(features,fake_labels)
    all_loader = DataLoader(all_dataset, batch_size=args.batch,shuffle=True)
    

       
    if args.cl_alg=="ggd":
        ### GGD method
        model = GGD(features.size(-1),
            args.hidden,
            args.layer,
            nn.PReLU(args.hidden),
            args.dropout,
            args.proj_layers)
    else:
        ### our method
        model = PGL(features.size(-1),
                args.hidden,
                args.layer,
                args.dropout,
                args.proj_layers)
    model.cuda()

    # evaluator = Evaluator(name='ogbn-papers100M')
    model_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    b_xent = nn.BCEWithLogitsLoss()
    
    # train deep graph infomax
    cnt_wait = 0
    best = 1e9
    best_t = 0
    counts = 0
    dur = []
    time_forward=0
    tag = str(int(np.random.random() * 10000000000))
    t_start = time.time()
    time_epoch=0
    for epoch in range(args.n_ggd_epochs):
        model.train()
        t0 = time.time()
        
        for step, (x,y) in enumerate(all_loader):
            model_optimizer.zero_grad()
            # aug_feat = aug_feature_dropout(x, args.drop_feat)
            t_st=time.time()
            loss = model(x.cuda(), y.cuda(), b_xent)
            # print("time_forward",time_forward)
            loss.backward()
            model_optimizer.step()
            time_epoch+=(time.time()-t_st)

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'pkl/best_ggd' + tag + '.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

        dur.append(time.time() - t0)
        print("time.time():", time.time())
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Time forward:{:.4f}  | "
            "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(), time_forward,
                                            m / np.mean(dur) / 1000))

        counts += 1
    pretrain_time = time_epoch
    print("pretraining time:", pretrain_time)
    del features_n
    gc.collect()
    print('Training Completed.')

    # return 1,2,3

    # create classifier model
    # classifier = Classifier(args.hidden, label_dim)

    # classifier = Classifier(features_train.size(-1), label_dim)
    
    # classifier.cuda()

    # classifier_optimizer = torch.optim.AdamW(classifier.parameters(),
    #                                         lr=args.classifier_lr,
    #                                         weight_decay=args.weight_decay)

    # train classifier
    print('Loading {}th epoch'.format(best_t))
    if args.use_gcl=="no":
        use_cl = False
        print(" without contractive learning!")
    else:
        use_cl = True
    model.load_state_dict(torch.load('pkl/best_ggd' + tag + '.pkl'))
    model.eval()
    embeds_list = []

    #graph power embedding reinforcement
    for step, (x, y) in enumerate(all_loader_ori):
        if use_cl:
            embed = model.embed(x.cuda())

        else:
            embed = x.cuda()

        embeds_list.append(embed)
    show_gpu()
    del model
    del features
    del features_p
    del model_optimizer
    gc.collect()
    
    embeds = torch.cat(embeds_list, dim = 0)
    # print(sys.getsizeof(embeds) / 1024 / 1024, 'MB')

    # embeds = torch.as_tensor(embeds_list)
    # embeds = torch.tensor( [item.cpu().detach().numpy() for item in embeds_list] )
    del embeds_list
    gc.collect()
    
    embeds = sk_prep.normalize(X=embeds.cpu().numpy(), norm="l2")

    embeds = torch.FloatTensor(embeds).cuda()
    
    # embeds = features
    # print("np.unique(labels)s: ", np.where(labels<5))
    torch.cuda.empty_cache()

    # Visualization
    # plot_index = np.where(labels.cpu()>36)
    
    # plot_index = torch.LongTensor(plot_index).squeeze()
    # plot_index = plot_index[:1000]
    # print("plot_index: ", plot_index.size())
    
    # tsne_plt(embeds[plot_index], labels[plot_index].cuda(), save_path = "./convert/trained.png")

    print("GPU used::",torch.cuda.memory_allocated()/1024/1024,"MB")

    show_gpu()
    if use_cl:
        classifier = ClassMLP(args.hidden,args.hidden,label_dim,args.layer,args.dropout).cuda()
    else:
        classifier = ClassMLP(feature_dim,args.hidden,label_dim,args.layer,args.dropout).cuda()
    ### Instant original method
    

    train_dataset = ExtendDataset(embeds[train_idx],train_labels)
    valid_dataset = ExtendDataset(embeds[val_idx],val_labels)
    test_dataset = ExtendDataset(embeds[test_idx],test_labels)
    
 

    
    # all_loader = DataLoader(all_dataset, batch_size=args.batch,shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch,shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    if snapshot>0:
        
        sub_train = np.where(change_node_list)[0]
        print("sub_train:", sub_train.shape)
        mini_train_dataset = ExtendDataset(embeds[sub_train],labels[sub_train].unsqueeze(1))
        mini_train_loader = DataLoader(mini_train_dataset, batch_size=args.batch,shuffle=True)

        memory = np.where(~change_node_list)[0]
        # Ensure that there are enough samples in memory to match sub_train
        if len(memory) >= len(sub_train):
            # Sample indices from memory
            memory = np.random.choice(memory, size=len(sub_train), replace=False)
            print("Sampled indices from memory:", memory.shape)
        else:
            print("Not enough elements in memory to match the size of sub_train.")

        mem_dataset = ExtendDataset(embeds[memory],labels[memory].unsqueeze(1))
        mem_loader = DataLoader(mem_dataset, batch_size=args.batch,shuffle=True)
    print(classifier)
    
    evaluator = Evaluator(name='ogbn-papers100M')
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    bad_counter = 0
    best = 0
    best_epoch = 0
    train_time = 0
    checkpt_file = 'pretrained/'+'snapshot'+str(snapshot)+'.pt'
    grad_checkpt_file = 'pretrained/'+'grad_snapshot'+str(snapshot)+'.pt'

    # #Initialize the model with the random parameters    
    # classifier.reset_parameters()

    #Initialize the last snapshot for initialization
    if snapshot > 0:
        checkpt_file_for_initial = 'pretrained/'+'snapshot'+str(snapshot-1)+'.pt'
        grad_checkpt_file_for_initial = 'pretrained/'+'grad_snapshot'+str(snapshot-1)+'.pt'
        print("Load the last snapshot for initialization:"+checkpt_file_for_initial)
        classifier.load_state_dict(torch.load(checkpt_file_for_initial))
        loaded_gradients = torch.load(grad_checkpt_file_for_initial)
    #    for param in classifier.parameters():
    #     print(param)
    #    print("loaded_gradients", loaded_gradients)
        # for n, p in loaded_gradients.items():
        #     print("loaded_gradients", p)

        ref_grad_vec = update_grad.grad_to_vector(loaded_gradients)
        torch.set_printoptions(precision=8)
    

    
    # print("current_grad_vec.shape", current_grad_vec)
    # Printing parameter values without their names
    
    
    print("--------------------------")
    print("Training...")
    if snapshot==0:
        for epoch in range(args.epochs):
            loss_tra,train_ep = train(classifier,args.dev,train_loader,classifier_optimizer)
            t_st=time.time()
            f1_val = validate(classifier, args.dev, valid_loader, evaluator)
            train_time+=train_ep
            if(epoch+1)%1 == 0:
                print(f'Epoch:{epoch+1:02d},'
                f'Train_loss:{loss_tra:.3f}',
                f'Valid_acc:{100*f1_val:.2f}%',
                f'Time_cost:{train_ep:.3f}/{train_time:.3f}')
            if f1_val > best:
                best = f1_val
                best_epoch = epoch+1
                t_st=time.time()
                torch.save(classifier.state_dict(), checkpt_file)

                # Example: save gradients as a torch file
                gradients = {}
                for name, parameter in classifier.named_parameters():
                    gradients[name] = parameter.grad.clone()  # Use `.clone()` to save a copy of the gradient tensor
                    # print("parameter.grad.clone()", parameter.grad.clone())
                torch.save(gradients, grad_checkpt_file)

                bad_counter = 0
            else:
                bad_counter += 1
            if bad_counter == args.patience:
                break
    
    # For incremental model test, to test whether the model of last moment can be used in the next moment (seems no)
    # if snapshot > 0:
    #    checkpt_file = 'pretrained/'+'snapshot'+str(snapshot-1)+'.pt'
    #    print("*****************************"+checkpt_file)
    if snapshot>0:
        test_acc, metric_macro, metric_micro, roc = test(classifier, args.dev, test_loader, evaluator,checkpt_file_for_initial)
        print(f"Train cost: {train_time:.2f}s")
        print('Load {}th epoch'.format(best_epoch))
        print("Checkpt_file: ", checkpt_file_for_initial)
        print(f"Test accuracy:{100*test_acc:.2f}%")
        print(f"metric_macro:{100*metric_macro:.2f}%")
        print(f"metric_micro:{100*metric_micro:.2f}%")
        for epoch in range(args.epochs):
            loss_tra,train_ep = train_incremental(classifier,args.dev,train_loader,classifier_optimizer,mini_train_loader)
            # loss_tra,train_ep = train(classifier,args.dev,train_loader,classifier_optimizer)
            t_st=time.time()
            f1_val = validate(classifier, args.dev, valid_loader, evaluator)
            train_time+=train_ep
            if(epoch+1)%1 == 0:
                print(f'Epoch:{epoch+1:02d},'
                f'Train_loss:{loss_tra:.3f}',
                f'Valid_acc:{100*f1_val:.2f}%',
                f'Time_cost:{train_ep:.3f}/{train_time:.3f}')
            if f1_val > best:
                best = f1_val
                best_epoch = epoch+1
                t_st=time.time()
                torch.save(classifier.state_dict(), checkpt_file)

                # Example: save gradients as a torch file
                gradients = {}
                for name, parameter in classifier.named_parameters():
                    gradients[name] = parameter.grad.clone()  # Use `.clone()` to save a copy of the gradient tensor
                    # print("parameter.grad.clone()", parameter.grad.clone())
                torch.save(gradients, grad_checkpt_file)

                bad_counter = 0
            else:
                bad_counter += 1
            if bad_counter == args.patience:
                break

          

        

    memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    test_acc, metric_macro, metric_micro, roc = test(classifier, args.dev, test_loader, evaluator,checkpt_file)
    print(f"Train cost: {train_time:.2f}s")
    print('Load {}th epoch'.format(best_epoch))
    print(f"Test accuracy:{100*test_acc:.2f}%")
    print(f"metric_macro:{100*metric_macro:.2f}%")
    print(f"metric_micro:{100*metric_micro:.2f}%")
    print(f"ROC:{100*roc:.2f}%")
    print(f"Memory: {memory / 2**20:.3f}GB")
    with open('accuracy_and_memory.txt', 'a') as f:
        print('Dataset:'+args.dataset+" Use gcl?"+str(use_cl), file=f)
        print(f"snapshot:{snapshot:.2f}  "+f"metric_macro:{100*metric_macro:.2f}%  "+f"metric_micro:{100*metric_micro:.2f}%  "+f"Memory: {memory / 2**20:.3f}GB",file=f)
    exit(0)     
    return metric_macro, metric_micro, pretrain_time
    # dur = []
    # best_acc, best_val_acc = 0, 0
    # print('Testing Phase ==== Please Wait.')
    # for epoch in range(args.n_classifier_epochs):
    #     classifier.train()
    #     t0 = time.time()

    #     classifier_optimizer.zero_grad()
    #     preds = classifier(embeds)
    #     # print("preds[train_idx]:", preds[train_idx].shape)
    #     # print("labels[train_idx]:", labels[train_idx].shape)
    #     loss = F.nll_loss(preds[train_idx], labels[train_idx])
    #     loss.backward()
    #     classifier_optimizer.step()

    #     dur.append(time.time() - t0)

    #     val_acc = evaluate(classifier, embeds, labels, val_idx)
    #     if epoch > 1000:
    #         if val_acc > best_val_acc:
    #             best_val_acc = val_acc
    #             test_acc = evaluate(classifier, embeds, labels, test_idx)
    #             if test_acc > best_acc:
    #                 best_acc = test_acc
    #     print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
    #           "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
    #                                         val_acc, m / np.mean(dur) / 1000))
    # print("Valid Accuracy {:.4f}".format(best_val_acc))

    # # best_acc = evaluate(classifier, embeds, labels, test_mask)
    # print("Test Accuracy {:.4f}".format(best_acc))




if __name__ == '__main__':
    main()
