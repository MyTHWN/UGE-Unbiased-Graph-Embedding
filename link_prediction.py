"""
Link Prediction Task: predicting the existence of an edge between two arbitrary nodes in a graph.
===========================================
-  Model: DGL-based graphsage and gat encoder (and many more)
-  Loss: cross entropy. You can modify the loss as you want
-  Metric: AUC
"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import time
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import argparse
import os
import pandas as pd
import random

from sklearn.metrics import roc_auc_score
import dgl.data

from models import GCN, GAT, SGC, GraphSAGE
from create_dataset import MyDataset
from predictors import DotLinkPredictor, MLPLinkPredictor
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, 
                    help='Random seed.')
parser.add_argument('--device', type=int, default=0, 
                    help='cuda')

parser.add_argument('--samplebyNode', type=str, default='no',
                    help='whether to sample edges by node')
parser.add_argument('--evalbyNode', type=str, default='no',
                    help='whether to evaluate edges by node')

parser.add_argument('--model', type=str, default='gat', 
                    choices=['gcn', 'gat', 'sgc'], help='model variant')
parser.add_argument('--loss', type=str, default='entropy', 
                    choices=['entropy', 'bpr'], help='loss function')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dim1', type=int, default=64,
                    help='Number of first layer hidden units.')
parser.add_argument('--dim2', type=int, default=16,
                    help='Number of second layer hidden units.')

parser.add_argument('--predictor', type=str, default='dot',
                    choices=['dot', 'mlp'], help='Predictor of the output layer')

parser.add_argument('--dataset', type=str, default='movielens')
parser.add_argument('--out_dir', type=str, default='./pokec_embeddings_lu2', 
                    help='Output embedding folder.')

parser.add_argument('--sensitive_attr', type=str, default='None', 
                    help='Sensitive attribute to be debiased.')
parser.add_argument('--debias_method', type=str, default='None', 
                    choices=['regularization', 'weighting', 'combined', 'None'], 
                    help='Debiasing method to apply.')
parser.add_argument('--reg_weight', type=float, default=0.1,
                    help='Weight for the debiasing regularization.')  

args = parser.parse_args()

if args.sensitive_attr == 'None':
    args.sensitive_attr = None
if args.debias_method == 'None':
    args.debias_method = None

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)
    
######################################################################
if args.evalbyNode == 'yes' and args.samplebyNode == 'no':
    print ('!!! If evalbyNode , then we must first samplebyNode!!!')
    raise AssertionError()
sample_by_node = True if args.samplebyNode == 'yes' else False
eval_by_node = True if args.evalbyNode == 'yes' else False

######################################################################
# Set up device and fix random seed
print ('==== Environment ====')
device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1) # limit cpu use
print ('  pytorch version: ', torch.__version__)
# print ('  device: ', torch.cuda.current_device())

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

######################################################################
# Load and construct data for link prediction task
if args.sensitive_attr:
    weight_file = 'debiasing_weights/{}_debias_weights.csv'.format(args.sensitive_attr)
    weights = pd.read_csv(weight_file)['debias_weights']
    weights = torch.tensor(weights)
else:
    weights = None

print ('==== Dataset ====')
if sample_by_node == False:
    graph, features, train_pos_g, train_neg_g, \
    test_pos_g, test_neg_g, train_weights  = \
    construct_link_prediction_data(
        data_type=args.dataset, device=device, debias=args.sensitive_attr, weights=weights)
else:
    graph, features, train_pos_g, train_neg_g, \
    test_pos_g, test_neg_g, \
    train_index_dict, test_index_dict, train_weights = \
    construct_link_prediction_data_nodewise(
        data_type=args.dataset, device=device, debias=args.sensitive_attr, weights=weights)

n_features = features.shape[1]

# Initialize embedding model
if args.model == 'gcn':
    model = GCN(graph, in_dim=n_features, hidden_dim=args.dim1, out_dim=args.dim2)
elif args.model == 'gat':
    model = GAT(graph, in_dim=n_features, hidden_dim=args.dim1//8, out_dim=args.dim2, num_heads=8)
elif args.model == 'sgc':
    model = SGC(graph, in_dim=n_features, hidden_dim=args.dim1, out_dim=args.dim2)
elif args.model == 'sage':
    model = GraphSAGE(graph, in_dim=n_features, hidden_dim=args.dim1, out_dim=args.dim2)
else:
    exit(f"unknown gcn model: {args.model}")

model = model.to(device)

# Initialize link predictor
if args.predictor == 'dot':
    pred = DotLinkPredictor()
else:
    pred = MLPLinkPredictor(args.dim2) 

pred = pred.to(device)

optimizer = torch.optim.Adam(itertools.chain(model.parameters(), 
                            pred.parameters()), lr=args.lr)

######################################################################
print ('==== Training ====')
# Training loop
dur = []
cur = time.time()
for e in range(args.epochs):
    model.train()
    # forward propagation on training set
    h = model(features)
    train_pos_score = pred(train_pos_g, h)
    train_neg_score = pred(train_neg_g, h)
    if args.loss == 'entropy':
        loss = compute_entropy_loss(train_pos_score, train_neg_score,
                                train_index_dict, train_weights,
                                device=device, byNode=False)
    elif args.loss == 'bpr':
        loss = compute_bpr_loss(train_pos_score, train_neg_score,
                                train_index_dict, train_weights,
                                device=device, byNode=False)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    dur.append(time.time() - cur)
    cur = time.time()

    if e % 20 == 0:
        # evaluation on test set
        model.eval()
        with torch.no_grad():
            test_pos_score = pred(test_pos_g, h)
            test_neg_score = pred(test_neg_g, h)
            test_auc, test_ndcg = \
                compute_metric(test_pos_score, test_neg_score,
                               test_index_dict,
                               device=device, byNode = eval_by_node)
            train_auc, train_ndcg = \
                compute_metric(train_pos_score, train_neg_score,
                               train_index_dict,
                               device=device, byNode = eval_by_node)

        print("Epoch {:05d} | Loss {:.4f} | Train AUC {:.4f} | Train NDCG {:.4f} | Test AUC {:.4f} | Test NDCG {:.4f} | Time {:.4f}".format(
              e, loss.item(), train_auc, train_ndcg, test_auc, test_ndcg, dur[-1]))

        ######################################################################
        # Save embedding
        embeddings = h.detach().cpu().numpy()
        path = '{}/{}_{}_{}_{}_{}_embedding.bin'.format(
                args.out_dir, args.dataset, args.model, args.loss, str(args.lr).replace('.',''), args.epochs)

        if args.debias_method:
            path = '{}/{}_{}_{}_{}_{}_{}_{}_{}_embedding.bin'.format(args.out_dir, args.dataset, args.debias_method,
                        args.sensitive_attr, args.model, args.loss, str(args.lr).replace('.',''), 
                        str(args.reg_weight).replace('.',''), args.epochs)
        else:
            path = '{}/{}_{}_{}_{}_{}_embedding.bin'.format(args.out_dir, args.dataset, 
                        args.model, args.loss, str(args.lr).replace('.',''), args.epochs)

        os.makedirs(args.out_dir, exist_ok=True)
        with open(path, "wb") as output_file:
            pkl.dump(embeddings, output_file)
        print ('==== saved {} embeddings to {} ===='.format(embeddings.shape, path))