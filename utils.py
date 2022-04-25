"""
Util functions
"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import scipy.sparse as sp
import tqdm
from collections import defaultdict

from sklearn.metrics import roc_auc_score, ndcg_score, f1_score, accuracy_score, recall_score
from data_loader import MyDataset


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size != k:
        raise ValueError('Ranking List length < k')    
    return np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    sort_r = sorted(r,reverse = True)
    idcg = dcg_at_k(sort_r, k)
    if not idcg:
        print('.', end=' ')
        return 0.
    return dcg_at_k(r, k) / idcg


def compute_bpr_loss(pos_score, neg_score, pos_weights):
    """Compute bpr loss for pairwise ranking
    """

    diff = (pos_score - neg_score)
    log_likelh = torch.log(1 / (1 + torch.exp(-diff))) * pos_weights

    return -torch.sum(log_likelh) / log_likelh.shape[0]


def compute_entropy_loss(pos_score, neg_score, pos_weights):
    """Compute cross entropy loss for link prediction
    """
    neg_weights = torch.ones(len(neg_score)).to(neg_score.device)
    weights = torch.cat([pos_weights, neg_weights])
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(neg_score.device)

    return F.binary_cross_entropy_with_logits(scores, labels, weights)


def compute_metric(pos_score, neg_score):
    """Compute AUC, NDCG metric for link prediction
    """

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
                      np.expand_dims(scores.cpu(), axis=0)) # super slow!
    
    return auc, ndcg


def construct_link_prediction_data_by_node(data_name='movielens'):
    """Construct train/test dataset by node for link prediction 
    
    Parameters
    ----------
    data_name :
        name of dataset
        
    Returns
    -------
    link_pred_data_dict:
        store all data objects required for link prediction task; dictionary
        {
            train_g :
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
            train_pos_weights:
                edge weights which are original 0/1 indicators or precomputed weights for weighting-based uge-w; torch tensor
            train_index_dict:
                index dictionary for negative training edges; dictionary: key=src node, value=index of dst node in train_neg_g
            test_index_dict:
                index dictionary for negative testing edges; dictionary: key=src node, value=index of dst node in test_neg_g
        }
    
    """

    dataset = MyDataset(data_name=data_name)
    
    graph = dataset[0]
    features = graph.ndata['feat']
    
    weights = graph.edata['weight'].numpy().tolist()
    
    # # Key place to include precomputed weighting for UGE-W
    # if not uge_w:  # use original 0/1 edge weights if do not include uge_w
    #     weights = torch.ones(graph.number_of_edges())
    # else:  # assign precomputed weights for weighting-based debiasing
    #     weights = graph.edata['weight'].numpy().tolist()
    #     print('Precomputed weights for weighting-based debiasing Loaded')

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
    
    # For each node, split its edge set for training and testing sets:
    # -  Randomly picks 10% of the edges in test set as positive examples
    # -  Leave the rest for the training set
    # -  Sample 20 times more negative examples in both sets
    neg_rate = 20
    test_rate = 0.1
    test_pos_u, test_pos_v = [], []
    test_neg_u, test_neg_v = [], []
    train_pos_u, train_pos_v = [], []
    train_neg_u, train_neg_v = [], []
    test_eids = []
    train_pos_weights = []
    train_index_dict = {}
    test_index_dict = {}
    pbar = tqdm.tqdm(edge_dict.items())
    for src_n, des_ns in pbar: 
        pbar.set_description("Splitting train/test edges by node")
        
        pos_des_ns = np.random.permutation(des_ns)
        candidate_negs = np.array(list(des_nodes - set(pos_des_ns)))
        all_neg_des_ns_idx = np.random.randint(low=0, high=len(candidate_negs), 
                                           size=(len(pos_des_ns), neg_rate))
        all_neg_des_ns = candidate_negs[all_neg_des_ns_idx]

        # split test/train while sampling neg
        test_pos_size = int(len(pos_des_ns) * test_rate)
        for n in range(len(pos_des_ns)):
            # for each pos edge, sample neg_rate neg edges
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

    print ('Finish constructing train/test set for link Prediction.')
    print ('  TrainPosEdges: ', len(train_pos_u))
    print ('  TrainNegEdges: ', len(train_neg_u))
    print ('  TestPosEdges: ', len(test_pos_u))
    print ('  TestNegEdges: ', len(test_neg_u))

    # Remove the edges in the test set from the original graph:
    # -  A subgraph will be created from the original graph by ``dgl.remove_edges``
    # print(test_eids)
    train_g = dgl.remove_edges(graph, test_eids)

    # Construct positive graph and negative graph
    # -  Positive graph consists of all the positive examples as edges
    # -  Negative graph consists of all the negative examples
    # -  Both contain the same set of nodes as the original graph
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=graph.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=graph.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=graph.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=graph.number_of_nodes())
    
    # put all data objects into a dict
    link_pred_data_dict = {
        'train_g': train_g,
        'features': features,
        'train_pos_g': train_pos_g,
        'train_neg_g': train_neg_g,
        'test_pos_g': test_pos_g,
        'test_neg_g': test_neg_g,
        'train_pos_weights': train_pos_weights,
        'train_index_dict': train_index_dict,
        'test_index_dict': test_index_dict
    }

    return link_pred_data_dict
