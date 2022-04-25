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
import warnings
import tqdm

from sklearn.metrics import roc_auc_score
import dgl.data

from graph_models import GCN, GAT, SGC, GraphSAGE, Node2vec
from predictors import DotLinkPredictor, MLPLinkPredictor
from utils import compute_bpr_loss, compute_entropy_loss, compute_metric, construct_link_prediction_data_by_node

from data_loader import MyDataset
from data_loader import SENSITIVE_ATTR_DICT  # predefined sensitive attributes for different datasets
from data_loader import DATA_FOLDER

from evaluate import eval_unbiasedness_pokec, eval_unbiasedness_movielens

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def arg_parse():
    parser = argparse.ArgumentParser(description='UGE')
    
    parser.add_argument('--dataset', type=str, default='pokec-z', choices=['movielens', 'pokec-z', 'pokec-n']) 
    parser.add_argument('--model', type=str, default='sage', choices=['gcn', 'gat', 'sgc', 'sage', 'node2vec'], help='gcn model')
    parser.add_argument('--debias_method', type=str, default='none', choices=['uge-r', 'uge-w', 'uge-c', 'none'], help='debiasing method to apply')
    parser.add_argument('--debias_attr', type=str, default='none', help='sensitive attribute to be debiased')
    parser.add_argument('--reg_weight', type=float, default=0.1, help='weight for the regularization based debiasing term')  
    
    parser.add_argument('--loss', type=str, default='entropy', choices=['entropy', 'bpr'], help='loss function')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 loss on parameters)')                
    parser.add_argument('--dim1', type=int, default=64, help='number of first layer hidden units')
    parser.add_argument('--dim2', type=int, default=16, help='number of second layer hidden units')
    parser.add_argument('--predictor', type=str, default='dot', choices=['dot', 'mlp'], help='predictor of the output layer')
    parser.add_argument('--seed', type=int, default=15, help='random seed')
    parser.add_argument('--device', type=int, default=0, help='cuda')
    parser.add_argument('--out_dir', type=str, default='./embeddings', help='Output embedding folder.')
    
    return parser.parse_args()


# check and set config for valid debiasing method
def config_debias(debias_method, debias_attr, dataset):
    # make sure debias_attr is supported by the predefined attr list
    if (debias_method != 'none') and (debias_attr not in SENSITIVE_ATTR_DICT[dataset]):
        if debias_attr == 'none':
            raise AssertionError('Please specify sensitive attribute for {} to debias'.format(debias_method))
        else:
            raise AssertionError('{} is not a predefined sensitive attr for {} dataset'.format(debias_attr, dataset))
        
    # if no debiasing, set debias_attr to none
    if debias_method == 'none':
        if debias_attr != 'none':
            warnings.warn('no debias method specified, debias_attr will be reset as none')
            debias_attr = 'none'
    
    # if weighting-based debiasing (uge-w/uge-c) is activated
    # change the data_name to load the precomputed edge weights when loading dataset
    if debias_method in ['uge-w', 'uge-c']:
        dataset = '{}_debias_{}'.format(dataset, debias_attr)
        
    return debias_method, debias_attr, dataset
        
    
def learn_embeddings(args):
    
    setup_seed(args.seed)
    
    # set up device 
    print ('==== Environment ====')
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)  # limit cpu use
    print ('  pytorch version: ', torch.__version__)
    
    # check if the parsed debiasing arguments are valid; config debiasing env
    debias_method, debias_attr, data_name = config_debias(args.debias_method, args.debias_attr, args.dataset)
    
    ######################################################################
    # load and construct data for link prediction task
    # ! if weighting-based debiasing method is triggered, train_weights will load precomputed weights for debiasing
    
    print ('==== Dataset {} ===='.format(args.dataset))
    
    link_pred_data_dict = construct_link_prediction_data_by_node(data_name=data_name)
    
    graph = link_pred_data_dict['train_g'].to(device)
    features = link_pred_data_dict['features'].to(device)
    train_pos_g = link_pred_data_dict['train_pos_g'].to(device)
    train_neg_g = link_pred_data_dict['train_neg_g'].to(device)
    test_pos_g = link_pred_data_dict['test_pos_g'].to(device)
    test_neg_g = link_pred_data_dict['test_neg_g'].to(device)
    train_weights = link_pred_data_dict['train_pos_weights'].to(device)
    train_index_dict = link_pred_data_dict['train_index_dict']
    test_index_dict = link_pred_data_dict['test_index_dict']

    n_features = features.shape[1]
        
    ######################################################################
    # Initialize embedding model
    
    print ('==== Embedding model {} + predictor {} ===='.format(args.model, args.predictor))
    
    if args.model == 'gcn':
        model = GCN(graph, in_dim=n_features, hidden_dim=args.dim1, out_dim=args.dim2)
    elif args.model == 'gat':
        model = GAT(graph, in_dim=n_features, hidden_dim=args.dim1//8, out_dim=args.dim2, num_heads=8)
    elif args.model == 'sgc':
        model = SGC(graph, in_dim=n_features, hidden_dim=args.dim1, out_dim=args.dim2)
    elif args.model == 'sage':
        model = GraphSAGE(graph, in_dim=n_features, hidden_dim=args.dim1, out_dim=args.dim2)
    elif args.model == 'node2vec':
        model = Node2vec(graph, in_dim=features.shape[0], out_dim=args.dim2)
    else:
        raise AssertionError(f"unknown gcn model: {args.model}")

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
    # group nodes by attribute combination to support uge-r and uge-c
    if debias_method in ['uge-r', 'uge-c']:
        print ('Grouping nodes by attribute combination to support {}...'.format(debias_method))
        
        attribute_list = SENSITIVE_ATTR_DICT[args.dataset]
        
        non_sens_attr_ls = [attr for attr in attribute_list if attr!=debias_attr]
        non_sens_attr_idx = [i for i in range(len(attribute_list)) if attribute_list[i]!=debias_attr]

        attribute_file = '{}/{}_node_attribute.csv'.format(DATA_FOLDER, args.dataset)
        node_attributes = pd.read_csv(attribute_file)

        attr_comb_groups = node_attributes.groupby(attribute_list)
        nobias_comb_groups = node_attributes.groupby(non_sens_attr_ls)

        attr_comb_groups_map = {tuple(group[1].iloc[0]):list(group[1].index) 
                                for group in attr_comb_groups}
        nobias_attr_comb_groups_map = {tuple(group[1].iloc[0][non_sens_attr_ls]):list(group[1].index) 
                                    for group in nobias_comb_groups}

        print ('Group finished.')
        print ('  attr_comb_group_num:', len(attr_comb_groups_map.keys()))
        print ('  nobias_attr_comb_group_num:', len(nobias_attr_comb_groups_map.keys()))
        
    ######################################################################
    # start training loop

    def map_tuple(x, index_ls):
        return tuple([x[idx] for idx in index_ls])

    def mem_eff_matmul_mean(mtx1, mtx2):
        mtx1_rows = list(mtx1.shape)[0]
        if mtx1_rows <= 1000:
            return torch.mean(torch.matmul(mtx1, mtx2))
        else:
            value_sum = 0
            for i in range(mtx1_rows // 1000):
                value_sum += torch.sum(torch.matmul(mtx1[i*1000:(i+1)*1000, :], mtx2))
            if mtx1_rows % 1000 != 0:
                value_sum += torch.sum(torch.matmul(mtx1[(i+1)*1000:, :], mtx2))
            return value_sum / (list(mtx1.shape)[0] * list(mtx2.shape)[1])
        
    print ('==== Training with {} debias method ===='.format(debias_method))

    dur = []
    cur = time.time()
    pbar = tqdm.tqdm(range(args.epochs))
    for e in pbar:
        pbar.set_description("learning node embedding")
        
        model.train()
        
        # forward propagation on training set
        if args.model == 'node2vec':
            input_x = torch.Tensor([i for i in range(features.shape[0])]).long().to(device)
        else:
            input_x = features
            
        h = model(input_x)

        train_pos_score = pred(train_pos_g, h)
        train_neg_score = pred(train_neg_g, h)

        if args.loss == 'entropy':
            loss = compute_entropy_loss(train_pos_score, train_neg_score, train_weights)
        elif args.loss == 'bpr':
            loss = compute_bpr_loss(train_pos_score, train_neg_score, train_weights)
        else:
            raise AssertionError(f"unknown loss: {args.loss}")

        # if regularization-based debiasing method is triggered, calc and add the regularization term
        if debias_method in ['uge-r', 'uge-c']:
            regu_loss = 0
            scr_groups = random.sample(list(attr_comb_groups_map.keys()), 100)  
            dst_groups = random.sample(list(attr_comb_groups_map.keys()), 100)
            nobias_scr_groups = [map_tuple(group, non_sens_attr_idx) for group in scr_groups]
            nobias_dst_groups = [map_tuple(group, non_sens_attr_idx) for group in dst_groups]

            for group_idx in range(len(scr_groups)):
                scr_group_nodes = attr_comb_groups_map[scr_groups[group_idx]]
                dsc_group_nodes = attr_comb_groups_map[dst_groups[group_idx]]
                scr_node_embs = h[scr_group_nodes]
                dsc_node_embs = h[dsc_group_nodes]
                aver_score = mem_eff_matmul_mean(scr_node_embs, dsc_node_embs.T)

                nobias_scr_group_nodes = nobias_attr_comb_groups_map[nobias_scr_groups[group_idx]]
                nobias_dsc_group_nodes = nobias_attr_comb_groups_map[nobias_dst_groups[group_idx]]
                nobias_scr_node_embs = h[nobias_scr_group_nodes]
                nobias_dsc_node_embs = h[nobias_dsc_group_nodes]
                nobias_aver_score = mem_eff_matmul_mean(nobias_scr_node_embs, nobias_dsc_node_embs.T)

                regu_loss += torch.square(aver_score - nobias_aver_score)

            loss += args.reg_weight * regu_loss / 100

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dur.append(time.time() - cur)
        cur = time.time()

        if e % 20 == 20:
            # evaluation on test set
            # print ('evaluating at epoch {}...'.format(e))
            
            model.eval()
            
            with torch.no_grad():
                test_pos_score = pred(test_pos_g, h)
                test_neg_score = pred(test_neg_g, h)
                test_auc, test_ndcg = compute_metric(test_pos_score, test_neg_score)
                train_auc, train_ndcg = compute_metric(train_pos_score, train_neg_score)

            print("-- Epoch {:05d} | Loss {:.4f} | Train AUC {:.4f} | Train NDCG {:.4f} | Test AUC {:.4f} | Test NDCG {:.4f} | Time {:.4f}".format(
                  e, loss.item(), train_auc, train_ndcg, test_auc, test_ndcg, dur[-1]))

        # Save learned embedding dynamically
        embeddings = h.detach().cpu().numpy()

        os.makedirs(args.out_dir, exist_ok=True)

        if debias_method == 'none':
            path = '{}/{}_{}_{}_{}_{}_embedding.bin'.format(
                args.out_dir, args.dataset, args.model, args.loss, str(args.lr), args.epochs)
        elif debias_method == 'uge-w':
            path = '{}/{}_{}_{}_{}_{}_{}_{}.bin'.format(
                args.out_dir, args.dataset, args.model, args.loss, str(args.lr), args.epochs,
                debias_method, debias_attr)
        else:
            path = '{}/{}_{}_{}_{}_{}_{}_{}_{}.bin'.format(
                args.out_dir, args.dataset, args.model, args.loss, str(args.lr), args.epochs,
                debias_method, debias_attr, str(args.reg_weight))

        with open(path, "wb") as output_file:
            pkl.dump(embeddings, output_file)
    
    print ('-- embeddings with shape {} is saved to {}'.format(embeddings.shape, path))
            
    return path
    

def eval_embeddings(args, out_embed_path):
    print ('==== Evaluate {} debias method on {} ===='.format(args.debias_method, args.debias_attr))
    
    if args.dataset == 'movielens':
        eval_unbiasedness_movielens(args.dataset, out_embed_path)
    else: # pokec
        eval_unbiasedness_pokec(args.dataset, out_embed_path)
        
        
if __name__ == '__main__':

    args = arg_parse()
    out_embed_path = learn_embeddings(args)
    
    # evaluate unbiasedness
    results = eval_embeddings(args, out_embed_path)
