import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import pickle as pk
import argparse
import pandas as pd
import os


def load_node_attributes(attribute_file):
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


def load_adjacency_matrix(file, M):
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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_random', type=str, default='true', 
                        help='whether to evaluate the base embeddings.')
    parser.add_argument('--embedding_path', type=str, default="./composite_embeddings", 
                        help='embedding path.')
    parser.add_argument('--dataset', type=str, default="pokec", 
                        help='dataset to evaluate.')
                    
    args = parser.parse_args()

    if args.dataset.split('_')[0] == 'pokec': M = 67796  # pokec
    elif args.dataset.split('_')[0] == 'pokecnofeat': M = 67796  # pokec
    elif args.dataset.split('_')[0] == 'pokec2': M = 66569  # pokec2
    else: raise Exception('Invalid dataset!')

    eval_random = True if args.eval_random == 'true' else False
    
    node_attributes, attr_groups = \
        load_node_attributes('mydata/{}_node_attribute.csv'.format(args.dataset.split('_')[0]))
    adj_mtx = load_adjacency_matrix('mydata/{}_edge.csv'.format(args.dataset.split('_')[0]), M)

    files = [f for f in os.listdir(args.embedding_path) 
            if os.path.isfile(os.path.join(args.embedding_path, f)) and args.dataset+'_' in f]
    files.sort()
    
    os.makedirs(args.embedding_path+'_fairness_results', exist_ok=True)

    for file in files:
        bias_attr = file.split('_')[2]

        if eval_random:
            X_random = np.random.rand(*(M,16))

        embedding = pk.load(open(os.path.join(args.embedding_path, file), 'rb'))
        X_unbiased = embedding

        output_path = os.path.join(args.embedding_path+'_fairness_results', file.split('.')[0]+'.txt')
        result_output = open(output_path, 'w')

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
            np.random.seed(2021)

            attr_values = list(attr_group.keys())
            num_attr_value = len(attr_values)
            comb_indices = np.ndindex(*(num_attr_value,num_attr_value))

            DP_list = []
            EO_list = []
            if eval_random:
                DP_list_random = []
                EO_list_random = []

            for comb_idx in comb_indices:
                group0 = attr_group[attr_values[comb_idx[0]]]
                group1 = attr_group[attr_values[comb_idx[1]]]
                group0_nodes = np.random.choice(group0, num_sample_pairs, replace=True)
                group1_nodes = np.random.choice(group1, num_sample_pairs, replace=True)

                pair_scores = np.sum(np.multiply(X_unbiased[group0_nodes], X_unbiased[group1_nodes]), axis=1)
                DP_prob = np.sum(sigmoid(pair_scores)) / num_sample_pairs
                # print(len(pair_scores), DP_prob)
                DP_list.append(DP_prob)

                comb_edge_indicator = (adj_mtx[group0_nodes, group1_nodes] > 0).astype(int)
                if np.sum(comb_edge_indicator) > 0:
                    EO_prob = np.sum(sigmoid(pair_scores.T[0]) * comb_edge_indicator) / np.sum(comb_edge_indicator)
                    EO_list.append(EO_prob)

                if eval_random:
                    pair_scores = np.sum(np.multiply(X_random[group0_nodes], X_random[group1_nodes]), axis=1)
                    DP_prob = np.sum(sigmoid(pair_scores)) / num_sample_pairs
                    DP_list_random.append(DP_prob)
                    if np.sum(comb_edge_indicator) > 0:
                        EO_prob = np.sum(sigmoid(pair_scores.T[0]) * comb_edge_indicator) / np.sum(comb_edge_indicator)
                        EO_list_random.append(EO_prob)
            
            DP_value = max(DP_list) - min(DP_list)
            EO_value = max(EO_list) - min(EO_list)
            if eval_random:
                DP_value_random = max(DP_list_random) - min(DP_list_random)
                EO_value_random = max(EO_list_random) - min(EO_list_random)

            print('sensitive_attribute debiased: ', bias_attr)
            result_output.write('sensitive_attribute debiased: ' + bias_attr + '\n')

            print('attribute evaluated: ', evaluate_attr)
            result_output.write('attribute evaluated: ' + evaluate_attr + '\n')
            
            print('DP_value: ', DP_value)
            result_output.write('DP_value: ' + str(DP_value) + '\n')
            print('EO_value: ', EO_value)
            result_output.write('EO_value: ' + str(EO_value) + '\n')

            if eval_random:
                print('DP_value_random: ', DP_value_random)
                result_output.write('DP_value_random: ' + str(DP_value_random) + '\n')
                print('EO_value_random: ', EO_value_random)
                result_output.write('EO_value_random: ' + str(EO_value_random) + '\n')

        result_output.close()