"""
Util functions
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
import tqdm
from collections import defaultdict

from sklearn.metrics import roc_auc_score, ndcg_score, f1_score, accuracy_score, recall_score

import dgl.data
from create_dataset import MyDataset

def compute_bpr_loss(pos_score, neg_score, index_dict, 
                     pos_weights, device='cpu', byNode=False):
    """Compute bpr loss for pairwise ranking
    """

    diff = (pos_score - neg_score)
    log_likelh = torch.log(1 / (1 + torch.exp(-diff))) * pos_weights

    # print(likelh.shape)
    return -torch.sum(log_likelh) / log_likelh.shape[0]

def compute_entropy_loss(pos_score, neg_score, index_dict, pos_weights, 
                        device='cpu', byNode=False):
    """Compute cross entropy loss for link prediction
    """
    neg_weights = torch.ones(len(neg_score)).to(device)
    weights = torch.cat([pos_weights, neg_weights]).to(device)
    scores = torch.cat([pos_score, neg_score]).to(device)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)
    return F.binary_cross_entropy_with_logits(scores, labels, weights)

def compute_metric(pos_score, neg_score, index_dict, device='cpu', byNode=False):
    """Compute AUC, NDCG metric for link prediction
    """
            
    if byNode == False:
        scores = torch.sigmoid(torch.cat([pos_score, neg_score])) # the probability of positive label
        scores_flip = 1.0 - scores # the probability of negative label
        y_pred =  torch.transpose(torch.stack((scores, scores_flip)), 0, 1)

        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
        labels_flip = 1 - labels # to generate one-hot labels
        y_true = torch.transpose(torch.stack((labels, labels_flip)), 0, 1).int()

        # print(y_true.cpu(), y_pred.cpu())
        auc = roc_auc_score(y_true.cpu(), y_pred.cpu())
        # ndcg = 0 
        ndcg = ndcg_score(np.expand_dims(labels.cpu(), axis=0), 
                          np.expand_dims(scores.cpu(), axis=0)) # super slow! you can comment it 
    else:
        auc = []
        ndcg = []
        for src_n, idxs in tqdm.tqdm(index_dict.items()):
            if len(idxs) == 0: # this may happen when not sample by node
                continue
            
            scores = torch.sigmoid(torch.cat([pos_score[idxs], neg_score[idxs]]))
            scores_flip = 1.0 - scores 
            y_pred =  torch.transpose(torch.stack((scores, scores_flip)), 0, 1)

            labels = torch.cat([torch.ones(pos_score[idxs].shape[0]), torch.zeros(neg_score[idxs].shape[0])]) 
            labels_flip = 1 - labels
            y_true = torch.transpose(torch.stack((labels, labels_flip)), 0, 1).int()

            auc.append(roc_auc_score(y_true.cpu(), y_pred.cpu()))
            ndcg.append(ndcg_score(np.expand_dims(labels.cpu(), axis=0), 
                          np.expand_dims(scores.cpu(), axis=0)))

        auc = np.mean(np.array(auc))
        ndcg = np.mean(np.array(ndcg))

    return auc, ndcg

def evaluate_acc(logits, labels, mask):
    """Compute Accuracy for node classification
    """

    logits = logits[mask]
    labels = labels[mask]

    _, indices = torch.max(logits, dim=1)

    acc = accuracy_score(labels, indices)
    recall = recall_score(labels, indices, average='micro')
    f1 = f1_score(labels, indices, average='micro')

    return acc, recall, f1

    # _, indices = torch.max(logits, dim=1)
    # correct =torch.sum(indices==labels)
    # return correct.item() * 1.0 / len(labels) # Accuracy

def load_node_classification_data(data_type='cora', device='cpu'):
    """Construct dataset for node classification
    
    Parameters
    ----------
    data_type :
        name of dataset
    Returns
    -------
    graph : 
        graph data; dgl.graph
    features :
        node feature; torch tensor
    labels :
        node label; torch tensor
    train/valid/test_mask :
        node mask for different set; torch tensor
    """

    if data_type == 'cora':
        dataset = dgl.data.CoraGraphDataset()
    elif data_type == 'citeseer':
        dataset = dgl.data.CiteseerGraphDataset()
    else:
        dataset = MyDataset(data_name=data_type)
    
    graph = dataset[0].to(device)
    features = graph.ndata['feat'].to(device)
    labels = graph.ndata['label'].to(device)
    train_mask = graph.ndata['train_mask'].to(device)
    valid_mask = graph.ndata['val_mask'].to(device)
    test_mask  = graph.ndata['test_mask'].to(device)
    return graph, features, labels, train_mask, valid_mask, test_mask

def construct_link_prediction_data(data_type='cora', device='cpu', 
                                    debias=None, weights=None):
    """Construct dataset for link prediction
    
    Parameters
    ----------
    data_type :
        name of dataset
    Returns
    -------
    train_graph :
        graph reconstructed by all training nodes; dgl.graph
    features :
        node feature; torch tensor
    train_pos_g :
        graph reconstructed by positive training edges; dgl.graph
    train_neg_g :
        graph reconstructed by negative training edges; dgl.graph
    test_pos_g :
        graph reconstructed by positive testing edges; dgl.graph
    test_neg_g :
        graph reconstructed by negative testing edges; dgl.graph
    """

    dataset = MyDataset(data_name=data_type)
    
    graph = dataset[0]
    features = graph.ndata['feat']

    # Split the edge set for training and testing sets:
    # -  Randomly picks 10% of the edges in test set as positive examples
    # -  Leave the rest for the training set
    # -  Sample the same number of edges for negative examples in both sets
    u, v = graph.edges()

    eids = np.arange(graph.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    train_size = graph.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
    print(f'  NumTrainLink: {train_size}')
    print(f'  NumTestLink: {test_size}')

    if not debias:
        weights = torch.ones(graph.number_of_edges())

    train_pos_weights = weights[eids[test_size:]]

    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    if data_type != 'movielens':
        mask = np.eye(adj.shape[0])
    else: # for bipartite graph, mask user-user pairs
        mask = np.zeros_like(adj.todense())
        mask[:adj.shape[0], :adj.shape[0]] = 1 # requirement: src node starts at 0
    adj_neg = 1 - adj.todense() - mask
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), graph.number_of_edges() // 2)
    # neg_eids = np.random.permutation(np.arange(len(neg_u))) # np.random.choice(len(neg_u), graph.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
        
    # tranform to tensor
    # test_pos_u, test_pos_v = torch.tensor(test_pos_u), torch.tensor(test_pos_v)
    test_neg_u, test_neg_v = torch.tensor(test_neg_u), torch.tensor(test_neg_v)
    # train_pos_u, train_pos_v = torch.tensor(train_pos_u), torch.tensor(train_pos_v)
    train_neg_u, train_neg_v = torch.tensor(train_neg_u), torch.tensor(train_neg_v)

    print ('==== Link Prediction Data ====')
    print ('  TrainPosEdges: ', len(train_pos_u))
    print ('  TrainNegEdges: ', len(train_neg_u))
    print ('  TestPosEdges: ', len(test_pos_u))
    print ('  TestNegEdges: ', len(test_neg_u))

    # Remove the edges in the test set from the original graph:
    # -  A subgraph will be created from the original graph by ``dgl.remove_edges``
    train_graph = dgl.remove_edges(graph, eids[:test_size])

    # Construct positive graph and negative graph
    # -  Positive graph consists of all the positive examples as edges
    # -  Negative graph consists of all the negative examples
    # -  Both contain the same set of nodes as the original graph
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=graph.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=graph.number_of_nodes())
    train_pos_weights = train_pos_weights.clone()

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=graph.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=graph.number_of_nodes())

    return train_graph.to(device), features.to(device), \
           train_pos_g.to(device), train_neg_g.to(device), \
           test_pos_g.to(device), test_neg_g.to(device), \
           train_pos_weights.to(device)

def construct_link_prediction_data_nodewise(data_type='cora', device='cpu',
                                            debias=None, weights=None):
    """Construct dataset for link prediction
    
    Parameters
    ----------
    data_type :
        name of dataset
    Returns
    -------
    train_graph :
        graph reconstructed by all training nodes; dgl.graph
    features :
        node feature; torch tensor
    train_pos_g :
        graph reconstructed by positive training edges; dgl.graph
    train_neg_g :
        graph reconstructed by negative training edges; dgl.graph
    test_pos_g :
        graph reconstructed by positive testing edges; dgl.graph
    test_neg_g :
        graph reconstructed by negative testing edges; dgl.graph
    """

    dataset = MyDataset(data_name=data_type)
    
    graph = dataset[0]
    features = graph.ndata['feat']

    # Split the edge set for training and testing sets:
    # -  Randomly picks 10% of the edges in test set as positive examples
    # -  Leave the rest for the training set
    # -  Sample the same number of edges for negative examples in both sets
    u, v, eids = graph.edges(form='all')

    # edges grouped by node
    src_nodes = set(u.numpy().tolist()) # all source node idx
    des_nodes = set(v.numpy().tolist()) # all destination node idx
    edge_dict = {}
    eid_dict = {}
    for i in range(int(len(u.numpy().tolist())/1)):
        if u.numpy()[i] not in edge_dict:
            edge_dict[u.numpy()[i]] = []
        edge_dict[u.numpy()[i]].append(v.numpy()[i])
        eid_dict[(u.numpy()[i], v.numpy()[i])] = eids.numpy()[i]
    
    # sample edges by node
    neg_rate = 20
    test_rate = 0.1
    test_pos_u, test_pos_v = [], []
    test_neg_u, test_neg_v = [], []
    train_pos_u, train_pos_v = [], []
    train_neg_u, train_neg_v = [], []
    test_eids = []
    train_pos_weights = []
    
    if not debias:
        weights = torch.ones(graph.number_of_edges())
    else:
        weights = graph.edata['weight'].numpy().tolist()
        print('loading weights for bias')

    train_index_dict = {}
    test_index_dict = {}
    cnt = 0
    for src_n, des_ns in tqdm.tqdm(edge_dict.items()): 
        cnt += 1
        # if cnt == 10: break   
        pos_des_ns = np.random.permutation(des_ns)
        candidate_negs = np.array(list(des_nodes - set(pos_des_ns)))
        all_neg_des_ns_idx = np.random.randint(low=0, high=len(candidate_negs), 
                                           size=(len(pos_des_ns), neg_rate))
        all_neg_des_ns = candidate_negs[all_neg_des_ns_idx]

        # split test/train while sampling neg
        test_pos_size = int(len(pos_des_ns) * test_rate)
        for n in range(len(pos_des_ns)):
            # for each pos edge, sample neg_rate neg edges
            # neg_des_ns = np.random.choice(candidate_negs, neg_rate)
            neg_des_ns = all_neg_des_ns[n]

            if n < test_pos_size: # testing set
                test_neg_v += list(neg_des_ns)
                test_neg_u += [src_n] * len(neg_des_ns)
                test_pos_v += [pos_des_ns[n]] * len(neg_des_ns)
                test_pos_u += [src_n] * len(neg_des_ns)
                test_eids.append(eid_dict[(src_n, pos_des_ns[n])])
                # store index grouped by node
                test_index_dict[src_n] = [len(test_neg_v)-1-i for i in range(len(neg_des_ns))]
            else: # training set
                train_neg_v += list(neg_des_ns)
                train_neg_u += [src_n] * len(neg_des_ns)
                train_pos_v += [pos_des_ns[n]] * len(neg_des_ns)
                train_pos_u += [src_n] * len(neg_des_ns)

                train_pos_weights += [weights[eid_dict[(src_n, pos_des_ns[n])]]] * len(neg_des_ns)
                # store index grouped by node
                train_index_dict[src_n] = [len(train_neg_v)-1-i for i in range(len(neg_des_ns))]
        
    # tranform to tensor
    test_pos_u, test_pos_v = torch.tensor(test_pos_u), torch.tensor(test_pos_v)
    test_neg_u, test_neg_v = torch.tensor(test_neg_u), torch.tensor(test_neg_v)
    train_pos_u, train_pos_v = torch.tensor(train_pos_u), torch.tensor(train_pos_v)
    train_neg_u, train_neg_v = torch.tensor(train_neg_u), torch.tensor(train_neg_v)
    test_eids = torch.tensor(test_eids)
    test_eids = test_eids.type(torch.int64)
    train_pos_weights = torch.tensor(train_pos_weights)

    print ('==== Link Prediction Data ====')
    print ('  TrainPosEdges: ', len(train_pos_u))
    print ('  TrainNegEdges: ', len(train_neg_u))
    print ('  TestPosEdges: ', len(test_pos_u))
    print ('  TestNegEdges: ', len(test_neg_u))

    # Remove the edges in the test set from the original graph:
    # -  A subgraph will be created from the original graph by ``dgl.remove_edges``
    # print(test_eids)
    train_graph = dgl.remove_edges(graph, test_eids)

    # Construct positive graph and negative graph
    # -  Positive graph consists of all the positive examples as edges
    # -  Negative graph consists of all the negative examples
    # -  Both contain the same set of nodes as the original graph
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=graph.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=graph.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=graph.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=graph.number_of_nodes())

    return train_graph.to(device), features.to(device), \
           train_pos_g.to(device), train_neg_g.to(device), \
           test_pos_g.to(device), test_neg_g.to(device), \
           train_index_dict, test_index_dict, train_pos_weights.to(device)


def construct_link_prediction_data_nodewise_fairwalk(
    data_type='cora', device='cpu', debias=None):
    """Construct dataset for link prediction
    
    Parameters
    ----------
    data_type :
        name of dataset
    Returns
    -------
    train_graph :
        graph reconstructed by all training nodes; dgl.graph
    features :
        node feature; torch tensor
    train_pos_g :
        graph reconstructed by positive training edges; dgl.graph
    train_neg_g :
        graph reconstructed by negative training edges; dgl.graph
    test_pos_g :
        graph reconstructed by positive testing edges; dgl.graph
    test_neg_g :
        graph reconstructed by negative testing edges; dgl.graph
    """

    dataset = MyDataset(data_name=data_type)
    
    graph = dataset[0]
    features = graph.ndata['feat']
    sensitive_attrs = graph.ndata[debias] if debias else None

    # Split the edge set for training and testing sets:
    # -  Randomly picks 10% of the edges in test set as positive examples
    # -  Leave the rest for the training set
    # -  Sample the same number of edges for negative examples in both sets
    u, v, eids = graph.edges(form='all')

    # edges grouped by node
    src_nodes = set(u.numpy().tolist()) # all source node idx
    des_nodes = set(v.numpy().tolist()) # all destination node idx
    edge_dict = {}
    eid_dict = {}
    for i in range(int(len(u.numpy().tolist())/1)):
        if u.numpy()[i] not in edge_dict:
            edge_dict[u.numpy()[i]] = []
        edge_dict[u.numpy()[i]].append(v.numpy()[i])
        eid_dict[(u.numpy()[i], v.numpy()[i])] = eids.numpy()[i]
    
    # sample edges by node
    neg_rate = 20
    test_rate = 0.1
    test_pos_u, test_pos_v = [], []
    test_neg_u, test_neg_v = [], []
    train_pos_u, train_pos_v = [], []
    train_neg_u, train_neg_v = [], []
    test_eids = []
    train_pos_weights = []
    
    if not debias:
        weights = torch.ones(graph.number_of_edges())
    else:
        weights = graph.edata['weight'].numpy().tolist()
        print('loading weights for bias')

    train_index_dict = {}
    test_index_dict = {}
    cnt = 0
    for src_n, des_ns in tqdm.tqdm(edge_dict.items()): 
        cnt += 1
        # if cnt == 10: break   
        if sensitive_attrs is None:  # normal random walk
            pos_des_ns = np.random.permutation(des_ns)
        else:  # fairwalk
            des_group = defaultdict(list)
            for n in range(len(des_ns)):
                des_group[sensitive_attrs[des_ns][n]].append(des_ns[n])
            num_des_per_group = int(len(des_ns)/len(des_group))  # num of node per group
            num_des_groups = [num_des_per_group for i in range(len(des_group)-1)] + [len(des_ns) -  (len(des_group)-1)*num_des_per_group]
            
            pos_des_ns = []
            groups = list(des_group.keys())
            for i in range(len(groups)):
                pos_des_ns += list(np.random.choice(des_group[groups[i]], num_des_groups[i]))
            
        candidate_negs = np.array(list(des_nodes - set(pos_des_ns)))
        all_neg_des_ns_idx = np.random.randint(low=0, high=len(candidate_negs), 
                                           size=(len(pos_des_ns), neg_rate))
        all_neg_des_ns = candidate_negs[all_neg_des_ns_idx]

        # split test/train while sampling neg
        test_pos_size = int(len(pos_des_ns) * test_rate)
        for n in range(len(pos_des_ns)):
            # for each pos edge, sample neg_rate neg edges
            # neg_des_ns = np.random.choice(candidate_negs, neg_rate)
            neg_des_ns = all_neg_des_ns[n]

            if n < test_pos_size: # testing set
                test_neg_v += list(neg_des_ns)
                test_neg_u += [src_n] * len(neg_des_ns)
                test_pos_v += [pos_des_ns[n]] * len(neg_des_ns)
                test_pos_u += [src_n] * len(neg_des_ns)
                test_eids.append(eid_dict[(src_n, pos_des_ns[n])])
                # store index grouped by node
                test_index_dict[src_n] = [len(test_neg_v)-1-i for i in range(len(neg_des_ns))]
            else: # training set
                train_neg_v += list(neg_des_ns)
                train_neg_u += [src_n] * len(neg_des_ns)
                train_pos_v += [pos_des_ns[n]] * len(neg_des_ns)
                train_pos_u += [src_n] * len(neg_des_ns)

                train_pos_weights += [weights[eid_dict[(src_n, pos_des_ns[n])]]] * len(neg_des_ns)
                # store index grouped by node
                train_index_dict[src_n] = [len(train_neg_v)-1-i for i in range(len(neg_des_ns))]
        
    # tranform to tensor
    test_pos_u, test_pos_v = torch.tensor(test_pos_u), torch.tensor(test_pos_v)
    test_neg_u, test_neg_v = torch.tensor(test_neg_u), torch.tensor(test_neg_v)
    train_pos_u, train_pos_v = torch.tensor(train_pos_u), torch.tensor(train_pos_v)
    train_neg_u, train_neg_v = torch.tensor(train_neg_u), torch.tensor(train_neg_v)
    test_eids = torch.tensor(test_eids)
    test_eids = test_eids.type(torch.int64)
    train_pos_weights = torch.tensor(train_pos_weights)

    print ('==== Link Prediction Data ====')
    print ('  TrainPosEdges: ', len(train_pos_u))
    print ('  TrainNegEdges: ', len(train_neg_u))
    print ('  TestPosEdges: ', len(test_pos_u))
    print ('  TestNegEdges: ', len(test_neg_u))

    # Remove the edges in the test set from the original graph:
    # -  A subgraph will be created from the original graph by ``dgl.remove_edges``
    # print(test_eids)
    train_graph = dgl.remove_edges(graph, test_eids)

    # Construct positive graph and negative graph
    # -  Positive graph consists of all the positive examples as edges
    # -  Negative graph consists of all the negative examples
    # -  Both contain the same set of nodes as the original graph
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=graph.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=graph.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=graph.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=graph.number_of_nodes())

    return train_graph.to(device), features.to(device), \
           train_pos_g.to(device), train_neg_g.to(device), \
           test_pos_g.to(device), test_neg_g.to(device), \
           train_index_dict, test_index_dict, train_pos_weights.to(device)
