import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import pickle as pk
import argparse
import pandas as pd
import os
import json

from utils import dcg_at_k, ndcg_at_k
from data_loader import SENSITIVE_ATTR_DICT  # predefined sensitive attributes for different datasets
from data_loader import DATA_FOLDER, RAW_FOLDER


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# load predefined sensitive attributes for pokec data
# csv format for each line: node_idx, gender, region, age
def load_node_attributes_pokec(attribute_file):
    node_attributes = {}
    gender_group = {}
    region_group = {}
    age_group = {}

    with open(attribute_file, 'r') as fin:
        lines = fin.readlines()
        idx = 0
        for line in lines:
            if idx == 0:
                idx += 1
                continue
            eachline = line.strip().split(',')
            node_idx = int(idx-1)
            gender = int(float(eachline[0]))
            region = int(eachline[1])
            age = int(float(eachline[2]))
            node_attributes[node_idx] = (gender, region, age)
            
            if gender not in gender_group:
                gender_group[gender] = []
            gender_group[gender].append(node_idx)

            if region not in region_group:
                region_group[region] = []
            region_group[region].append(node_idx)

            if age not in age_group:
                age_group[age] = []
            age_group[age].append(node_idx)

            idx += 1

    return node_attributes, {'gender':gender_group, 'region':region_group, 'age':age_group}


# load edge for pokec data
# csv format: src_node, dst_node, edge_weight (binary: 1.0)
def load_adjacency_matrix_pokec(file, M):
    adj_mtx = np.zeros((M,M))
    # load the adjacency matrix of size MxM of training or testing set
    print('loading data ...')
    with open(file, 'r') as fin:
        lines = fin.readlines()
        idx = 0 
        for line in lines:
            if idx == 0: 
                idx += 1
                continue
            eachline = line.strip().split(',')
            scr_node = int(eachline[0])
            dst_node = int(eachline[1])
            weight = float(eachline[2])
            adj_mtx[scr_node, dst_node] = weight 
            adj_mtx[dst_node, scr_node] = weight 
            idx += 1
    
    return adj_mtx


# load sensitive attributes for movielens
# dat format: user_idx, gender, age, occupation
def load_user_attributes_movielens(file, M):
    #(gender, age, occupation)
    user_attributes = {}
    gender_dist = {'F':0, 'M':0}
    age_dist = {1:0, 18:0, 25:0, 35:0, 45:0, 50:0, 56:0}
    occupation_dist = {occup:0 for occup in range(21)}

    with open(file, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            eachline = line.strip().split('::')
            user_idx = int(eachline[0])
            gender = eachline[1]
            age = int(eachline[2])
            occupation = int(eachline[3])
            user_attributes[user_idx] = (gender, age, occupation)

    return user_attributes


# load edges for movielens
# dat format: user_idx, item_idx, rating
def load_rating_matrix_movielens(file, M, N):
    over_rating_sparse_mtx = {}
    over_rating_mtx = np.zeros((M,N))
    #load the overall rating matrices of size MxN of training or testing set
    print('loading data ...')
    with open(file, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            eachline = line.strip().split('::')
            user_idx = int(eachline[0])
            item_idx = int(eachline[1])
            rating = int(eachline[2])
            over_rating_mtx[user_idx, item_idx] = rating
            over_rating_sparse_mtx[(user_idx, item_idx)] = rating
    
    return over_rating_sparse_mtx, over_rating_mtx


def eval_unbiasedness_pokec(data_name, embed_file=None):  # if file is none, then evalueate random embedding
    
    if data_name == 'pokec-z': M = 67796 
    elif data_name == 'pokec-n': M = 66569 
    else: raise Exception('Invalid dataset!')
    
    node_attributes, attr_groups = load_node_attributes_pokec('{}/{}_node_attribute.csv'.format(DATA_FOLDER, data_name))
    adj_mtx = load_adjacency_matrix_pokec('{}/{}_edge.csv'.format(DATA_FOLDER, data_name), M)
    
    genders = np.array([node_attributes[i][0] for i in range(M)])
    regions = np.array([node_attributes[i][1] for i in range(M)])
    ages = np.array([node_attributes[i][2] for i in range(M)])
    
    attribute_labels = {'gender': genders, 'age': ages, 'region': regions}
    
    
    if embed_file:  # learned embeddings loaded from valid file
        embedding = pk.load(open(embed_file, 'rb'))
    else: # random embeddings
        embedding = np.random.rand(*(M,16))
    
    results = {
        'unbiasedness': {
            'gender': 0.0, 
            'age': 0.0, 
            'region': 0.0
        },
        'fairness-DP':{
            'gender': 0.0, 
            'age': 0.0, 
            'region': 0.0
        },
        'fairness-EO':{
            'gender': 0.0, 
            'age': 0.0, 
            'region': 0.0
        },
        'utility': 0.0
    }
    
    # eval micro-f1 for attribute prediction (unbiasedness)
    print('Unbiasedness evaluation (predicting attribute)')
    for evaluate_attr in ['gender', 'age', 'region']:
                
        # eval learned embedding
        lgreg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=500).fit(
            embedding[:50000], attribute_labels[evaluate_attr][:50000])
        pred = lgreg.predict(embedding[50000:])
        
        score = f1_score(attribute_labels[evaluate_attr][50000:], pred, average='micro')
        
        print('-- micro-f1 when predicting {}: {.6f}'.format(evaluate_attr, score))
        results['unbiasedness'][evaluate_attr] = score

        
    # eval fairness (DP & EO)
    print('Fairness evaluation (DP & EO)')
    for evaluate_attr in ['gender', 'age', 'region']:
        
        if evaluate_attr == 'age': 
            num_sample_pairs = 200000
            attr_group = attr_groups[evaluate_attr]
            keys = list(attr_group.keys())
            for key in keys:
                if len(attr_group[key]) < 1000:
                    del attr_group[key]
        else:
            num_sample_pairs = 1000000
            attr_group = attr_groups[evaluate_attr]
            
        attr_values = list(attr_group.keys())
        num_attr_value = len(attr_values)
        comb_indices = np.ndindex(*(num_attr_value,num_attr_value))

        DP_list = []
        EO_list = []

        for comb_idx in comb_indices:
            group0 = attr_group[attr_values[comb_idx[0]]]
            group1 = attr_group[attr_values[comb_idx[1]]]
            group0_nodes = np.random.choice(group0, num_sample_pairs, replace=True)
            group1_nodes = np.random.choice(group1, num_sample_pairs, replace=True)

            pair_scores = np.sum(np.multiply(embedding[group0_nodes], embedding[group1_nodes]), axis=1)
            DP_prob = np.sum(sigmoid(pair_scores)) / num_sample_pairs
            DP_list.append(DP_prob)

            comb_edge_indicator = (adj_mtx[group0_nodes, group1_nodes] > 0).astype(int)
            if np.sum(comb_edge_indicator) > 0:
                EO_prob = np.sum(sigmoid(pair_scores.T[0]) * comb_edge_indicator) / np.sum(comb_edge_indicator)
                EO_list.append(EO_prob)
        
        DP_value = max(DP_list) - min(DP_list)
        EO_value = max(EO_list) - min(EO_list)
        
        print('-- DP_value when predicting {}: {.6f}'.format(evaluate_attr, DP_value))
        print('-- EO_value when predicting {}: {.6f}'.format(evaluate_attr, EO_value))
        results['fairness-DP'][evaluate_attr] = DP_value
        results['fairness-EO'][evaluate_attr] = EO_value
            
        
    #evaluate NDCG for link prediction
    k = 10
    accum_ndcg = 0
    node_cnt = 0

    print('Utility evaluation (link prediction)')
    for node_id in range(M):
        node_edges = adj_mtx[node_id]
        pos_nodes = np.where(node_edges>0)[0]
        # num_pos = len(pos_nodes)
        num_test_pos = int(len(pos_nodes) / 10) + 1
        test_pos_nodes = pos_nodes[:num_test_pos]
        num_pos = len(test_pos_nodes)

        if num_pos == 0 or num_pos >= 100:
            continue
        neg_nodes = np.random.choice(np.where(node_edges == 0)[0], 100-num_pos, replace=False)
        all_eval_nodes = np.concatenate((test_pos_nodes, neg_nodes)) 
        all_eval_edges = np.zeros(100)
        all_eval_edges[:num_pos] = 1

        pred_edges = np.dot(embedding[node_id], embedding[all_eval_nodes].T)
        rank_pred_keys = np.argsort(pred_edges)[::-1]
        ranked_node_edges = all_eval_edges[rank_pred_keys]
        ndcg = ndcg_at_k(ranked_node_edges, k)
        accum_ndcg += ndcg

        node_cnt += 1

    score = accum_ndcg/node_cnt
    print('-- ndcg of link prediction: {.6f}'.format(score))
    results['utility'] = score
    
    return results


def eval_unbiasedness_movielens(data_name, embed_file=None):
    
    M = 6040 + 1
    N = 3952 + 1
        
    rating_sparse_mtx, rating_mtx = load_rating_matrix_movielens('{}/ml-1m/ratings.dat'.format(RAW_FOLDER), M, N)
    user_attributes = load_user_attributes_movielens('{}/ml-1m/users.dat'.format(RAW_FOLDER), M)

    genders = np.array([int(user_attributes[i][0]=='M') for i in range(1, M)])
    ages = np.array([int(user_attributes[i][1]) for i in range(1, M)])
    occupations = np.array([int(user_attributes[i][2]) for i in range(1, M)])

    attribute_labels = {'gender': genders, 'age': ages, 'occupation': occupations}

    rating_mtx = rating_mtx[1:]
    rating_mtx = rating_mtx[:,1:]
    
    if embed_file:
        unbiased_embedding = pk.load(open(embed_file,'rb')) 
        X = unbiased_embedding[:M-1]  # users
        Y = unbiased_embedding[M-1:]  # items
    else:
        X, Y = np.random.rand(*(M-1,16)), np.random.rand(*(N-1,16))
    
    results = {
        'unbiasedness': {
            'gender': 0.0, 
            'age': 0.0, 
            'region': 0.0
        },
        # 'fairness-DP':{
        #     'gender': 0.0, 
        #     'age': 0.0, 
        #     'region': 0.0
        # },
        # 'fairness-EO':{
        #     'gender': 0.0, 
        #     'age': 0.0, 
        #     'region': 0.0
        # },
        'utility': 0.0
    }

    # eval micro-f1 for attribute prediction (unbiasedness)
    print('Unbiasedness evaluation (predicting attribute)')
    for evaluate_attr in ['gender', 'age', 'occupation']:
        
        lgreg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000).fit(
            X[:5000], attribute_labels[evaluate_attr][:5000])
        pred = lgreg.predict(X[5000:])

        # rating_lgreg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000).fit(
        #     rating_mtx[:5000], attribute_labels[evaluate_attr][:5000])
        # rating_pred = rating_lgreg.predict(rating_mtx[5000:])
        
        score = f1_score(attribute_labels[evaluate_attr][5000:], pred, average='micro')
        
        print('-- micro-f1 when predicting {}: {.6f}'.format(evaluate_attr, score))
        results['unbiasedness'][evaluate_attr] = score
        
        # score = f1_score(attribute_labels[evaluate_attr][5000:], rating_pred, average='micro')
        # print('-- raw rating micro-f1: ', score)


    #evaluate NDCG
    k = 10
    accum_ndcg = 0
    
    print('Utility evaluation (link prediction)')
    for user_id in range(1, M):
        user = user_id - 1
        user_ratings = rating_mtx[user] # (rating_mtx[user] > 0).astype(int)
        
        pred_ratings = np.dot(X[user], Y.T)
        rank_pred_keys = np.argsort(pred_ratings)[::-1]
        ranked_user_ratings = user_ratings[rank_pred_keys]
        ndcg = ndcg_at_k(ranked_user_ratings, k)
        accum_ndcg += ndcg
        
    score = accum_ndcg/M
    print('-- ndcg of link prediction: {.6f}'.format(score))
    results['utility'] = score

    return results


# evalute embeddings by batch of files
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--embedding_folder', type=str, default="./embeddings", help='embedding folder path')
    parser.add_argument('--dataset', type=str, default="pokec-z", help='dataset to evaluate')

    args = parser.parse_args()

    # evaluate the performance of all embeddings under a folder
    files = [f for f in os.listdir(args.embedding_folder) 
            if os.path.isfile(os.path.join(args.embedding_folder, f)) and args.dataset+'_' in f]
    files.sort()
    
    output_folder = args.embedding_folder + '_eval_results'
    os.makedirs(output_folder, exist_ok=True)

    for file in files:
        embed_file = os.path.join(args.embedding_folder, file)
        
        print ('=== Evaluating {} ==='.format(embed_file))
        
        if args.dataset == 'movielens':
            results = eval_unbiasedness_movielens(args.dataset, embed_file)
        else: # pokec
            results = eval_unbiasedness_pokec(args.dataset,  embed_file)
        
        output_path = os.path.join(output_folder, file.split('.')[0]+'.txt')
        with open(output_path, 'w') as fp:
            json.dump(results, fp)
