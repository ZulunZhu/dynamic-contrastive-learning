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
from model import ClassMLP,PGL,Classifier,GGD
from utils import *
from glob import glob
from tqdm import tqdm
from sklearn import preprocessing as sk_prep
import logging
import copy
import os
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
    
    parser.add_argument('--epsilon', type=float, default=8, help='epsilon.')
    parser.add_argument("--n-ggd-epochs", type=int, default=1,
                        help="number of training epochs")
    parser.add_argument("--n-classifier-epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--drop_feat", type=float, default=0.1,
                        help="feature dropout rate")
    # Learining parameters
    
    parser.add_argument("--classifier-lr", type=float, default=0.05, help="classifier learning rate")
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay.')
    parser.add_argument('--layer', type=int, default=4, help='number of layers.')
    parser.add_argument('--hidden', type=int, default=2048, help='hidden dimensions.')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate.')
    parser.add_argument('--bias', default='none', help='bias.')
    parser.add_argument("--proj_layers", type=int, default=1, help="number of project linear layers")
    parser.add_argument('--epochs', type=int, default= 200, help='number of epochs.')
    parser.add_argument('--batch', type=int, default=10000, help='batch size.')
    parser.add_argument("--patience", type=int, default=50, help="early stop patience condition")
    parser.add_argument('--dev', type=int, default=0, help='device id.')
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

    
    n,m,features,features_n,train_labels,val_labels,test_labels,train_idx,val_idx,test_idx,memory_dataset, py_alg = load_ogb_init(args.dataset, args.alpha,args.rmax, args.epsilon, args.alg) ##
    print("features::",features)
    
    features_p = features
    print('------------------ Initial -------------------')
    # print("train_idx:",train_idx.size())
    # print("val_idx:",val_idx.size())
    # print("test_idx:",test_idx.size())

    # prepare_to_train(0,features,features_p,features_n, m, train_idx, val_idx, test_idx, train_labels, val_labels, test_labels, args, checkpt_file)
    print('------------------ update -------------------')
    snapList = [f for f in glob('../data/'+args.dataset+'/*Edgeupdate_snap*.txt')]
    print('number of snapshots: ', len(snapList))
    # print("features[36][36]::",features[36][36])
    for i in range(len(snapList)):
        change_node_list = np.zeros([n]) 
        features_copy = copy.deepcopy(features)
        # print("change_node_list",np.sum(change_node_list))
        py_alg.snapshot_lazy('../data/'+args.dataset+'/'+args.dataset+'_Edgeupdate_snap'+str(i+1)+'.txt', args.rmax, args.alpha, features, features_p, change_node_list, args.alg)
        print("number of changed node", np.sum(change_node_list))
        # print("features_ori:",features)
        change_node_list = change_node_list.astype(bool)
        print("change_node_list:",change_node_list.shape)
        features_p = features_copy[~change_node_list]
        features_n = features_copy[change_node_list]
        print("feature_p.size:",features_p.shape)
        print("feature_n.size:",features_n.shape)
        prepare_to_train(i+1,features, features_p, features_n, m,train_idx, val_idx, test_idx, train_labels, val_labels, test_labels, args, checkpt_file)

def train(model, device, train_loader, optimizer):
    model.train()

    time_epoch=0
    loss_list=[]

    for step, (x, x_n, y) in enumerate(train_loader):
        t_st=time.time()
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out, y.squeeze(1))
        loss.backward()
        optimizer.step()
        time_epoch+=(time.time()-t_st)
        loss_list.append(loss.item())
        
    return np.mean(loss_list), time_epoch


@torch.no_grad()
def validate(model, device, loader, evaluator):
    model.eval()
    y_pred, y_true = [], []
    for step,(x, x_n, y) in enumerate(loader):
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
    for step,(x, x_n, y) in enumerate(loader):
        x = x.cuda()
        out = model(x)
        y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
        y_true.append(y)
    return evaluator.eval({
        "y_true": torch.cat(y_true, dim=0),
        "y_pred": torch.cat(y_pred, dim=0),
    })['acc']

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

def prepare_to_train(snapshot, features_ori, features_p, features_n,m, train_idx, val_idx, test_idx, train_labels, val_labels, test_labels, args, checkpt_file):
    features_ori = torch.FloatTensor(features_ori).cuda()
    features_n = torch.FloatTensor(features_n).cuda()
    features_p = torch.FloatTensor(features_p).cuda()
    n = features_ori.size()[0]
    feature_dim = features_ori.size(-1)
    print("Original feature size: ", features_ori.size(0))
    print("train_idx:", train_idx)
    print("n=", n, " m=", m)

    if(snapshot>0):    
       features = torch.cat((features_ori,features_p),dim=0)
    #    features_n = torch.cat((features_n,features_n[:features.size(0)-features_ori.size(0), :]),dim=0)## Not finished yet
    #    print("Added positive feature size: ", features.size(0)-features_ori.size(0))
    #    assert features.size(0)==features_n.size(0)
       
    else:
        features = features_ori




    label_dim = int(max(train_labels.max(),val_labels.max(),test_labels.max()))+1
    labels = torch.cat((train_labels, val_labels,test_labels)).squeeze(1).cuda()
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

    all_dataset_ori = ExtendDataset(features_ori,labels)
    all_loader_ori = DataLoader(all_dataset_ori, batch_size=args.batch,shuffle=False)

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

    tag = str(int(np.random.random() * 10000000000))

    for epoch in range(args.n_ggd_epochs):
        model.train()
        t0 = time.time()

        for step, (x,y) in enumerate(all_loader):
            model_optimizer.zero_grad()
            # aug_feat = aug_feature_dropout(x, args.drop_feat)
            loss = model(x.cuda(), y.cuda(), b_xent)
            loss.backward()
            model_optimizer.step()

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

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            m / np.mean(dur) / 1000))

        counts += 1
    del features_n
    gc.collect()
    print('Training Completed.')

    # create classifier model
    # classifier = Classifier(args.hidden, label_dim)

    # classifier = Classifier(features_train.size(-1), label_dim)
    
    # classifier.cuda()

    # classifier_optimizer = torch.optim.AdamW(classifier.parameters(),
    #                                         lr=args.classifier_lr,
    #                                         weight_decay=args.weight_decay)

    # train classifier
    print('Loading {}th epoch'.format(best_t))
    use_cl = True
    model.load_state_dict(torch.load('pkl/best_ggd' + tag + '.pkl'))
    model.eval()
    embeds = []
    #graph power embedding reinforcement
    for step, (x, y) in enumerate(all_loader_ori):
        if use_cl:
            embed = model.embed(x)
        else:
            embed = x

        embeds.append(embed)
    embeds = torch.cat(embeds, dim = 0)
    torch.cuda.empty_cache()
    embeds = sk_prep.normalize(X=embeds.cpu().numpy(), norm="l2")

    embeds = torch.FloatTensor(embeds).cuda()
    del model
    del features
    del features_p
    gc.collect()
    # embeds = features
    print("embeds: ", embeds)
    


    ### Instant original method
    train_dataset = SimpleDataset(embeds[train_idx],embeds[train_idx],train_labels)
    valid_dataset = SimpleDataset(embeds[val_idx],embeds[val_idx],val_labels)
    test_dataset = SimpleDataset(embeds[test_idx], embeds[test_idx],test_labels)

    # all_loader = DataLoader(all_dataset, batch_size=args.batch,shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch,shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    
    if use_cl:
        classifier = ClassMLP(args.hidden,args.hidden,label_dim,args.layer,args.dropout).cuda()
    else:
        classifier = ClassMLP(feature_dim,args.hidden,label_dim,args.layer,args.dropout).cuda()
    evaluator = Evaluator(name='ogbn-papers100M')
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    bad_counter = 0
    best = 0
    best_epoch = 0
    train_time = 0
    classifier.reset_parameters()
    print("--------------------------")
    print("Training...")
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
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter == args.patience:
            break

    test_acc = test(classifier, args.dev, test_loader, evaluator,checkpt_file)
    print(f"Train cost: {train_time:.2f}s")
    print('Load {}th epoch'.format(best_epoch))
    print(f"Test accuracy:{100*test_acc:.2f}%")



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
