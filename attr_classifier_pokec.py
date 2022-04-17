
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import pickle as pk
import argparse
import pandas as pd

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


def load_node_attributes(attribute_file):
	node_attributes = {}
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
			idx += 1

	return node_attributes


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


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--debias', type=str, default="gender", 
                    	choices=['gender', 'region', 'AGE'],
						help='attribute to debias.')
	parser.add_argument('--debias_method', type=str, default="weighting", 
                    	choices=['weighting', 'regularization', 'combined'],
						help='attribute to debias.')
	parser.add_argument('--dataset', type=str, default="pokec", 
						help='dataset to evaluate.')
	parser.add_argument('--model', type=str, default="gat", 
                    	choices=['gat', 'gcn', 'sgc'],
						help='embedding model.')
	parser.add_argument('--eval_base', type=str, default='true', 
						help='whether to evaluate the base embeddings.')
	parser.add_argument('--reg_weight', type=str, default='01', 
						help='weight for regularization.')					
	args = parser.parse_args()

	if args.dataset.split('_')[0] == 'pokec': M = 67796  # pokec
	elif args.dataset.split('_')[0] == 'pokecnofeat': M = 67796  # pokec
	elif args.dataset.split('_')[0] == 'pokec2': M = 66569  # pokec2
	else: raise Exception('Invalid dataset!')

	eval_base = True if args.eval_base == 'true' else False
	
	bias_attr = args.debias
	evaluate_attr = 'gender'

	node_attributes = load_node_attributes('mydata/{}_node_attribute.csv'.format(args.dataset.split('_')[0]))
	adj_mtx = load_adjacency_matrix('mydata/{}_edge.csv'.format(args.dataset.split('_')[0]), M)

	genders = np.array([node_attributes[i][0] for i in range(M)])
	regions = np.array([node_attributes[i][1] for i in range(M)])
	ages = np.array([node_attributes[i][2] for i in range(M)])

	attribute_labels = {'gender': genders, 'age': ages, 'region': regions}

	if eval_base:
		X_random = np.random.rand(*(M,16))
		biased_embedding = pk.load(open(
			'./pokec_embeddings_lu2/{}_{}_entropy_001_800_embedding.bin'.format(args.dataset.split('_')[0], args.model),'rb'))
		X_biased = biased_embedding

	if args.debias_method == 'regularization':
		dataset_name = args.dataset.split('_')[0]
	else: 
		dataset_name = args.dataset

	unbiased_embedding = pk.load(open(
	'./pokec_embeddings_lu2/{}_{}_{}_{}_entropy_001_{}_800_embedding.bin'.format(dataset_name, args.debias_method, args.debias, args.model, args.reg_weight),'rb'))
	X_unbiased = unbiased_embedding

	result_output = open('./results/{}_{}_{}_{}_entropy_001_{}_800_embedding.txt'.format(dataset_name, args.debias_method, args.debias, args.model, args.reg_weight), 'w')

	for evaluate_attr in ['gender', 'age', 'region']:
		if eval_base:
			bias_lgreg = LogisticRegression(random_state=0, class_weight='balanced', 
				max_iter=500).fit(X_biased[:50000], attribute_labels[evaluate_attr][:50000])
			base_pred = bias_lgreg.predict(X_biased[50000:])
			random_lgreg = LogisticRegression(random_state=0, class_weight='balanced', 
				max_iter=500).fit(X_random[:50000], attribute_labels[evaluate_attr][:50000])
			random_pred = random_lgreg.predict(X_random[50000:])
		
		unbias_lgreg = LogisticRegression(random_state=0, class_weight='balanced', 
			max_iter=500).fit(X_unbiased[:50000], attribute_labels[evaluate_attr][:50000])
		unbias_pred = unbias_lgreg.predict(X_unbiased[50000:])

		print('sensitive_attribute debiased: ', bias_attr)
		result_output.write('sensitive_attribute debiased: ' + bias_attr + '\n')

		print('attribute evaluated: ', evaluate_attr)
		result_output.write('attribute evaluated: ' + evaluate_attr + '\n')
		if eval_base:
			score = f1_score(attribute_labels[evaluate_attr][50000:], base_pred, average='micro')
			print('baseline micro-f1: ', score)
			result_output.write('baseline micro-f1: ' + str(score) + '\n')
		
		score = f1_score(attribute_labels[evaluate_attr][50000:], unbias_pred, average='micro')
		print('debiased micro-f1: ', score)
		result_output.write('debiased micro-f1: ' + str(score) + '\n')

		if eval_base:
			score = f1_score(attribute_labels[evaluate_attr][50000:], random_pred, average='micro')
			print('random embedding micro-f1: ', score)
			result_output.write('random embedding micro-f1: ' + str(score) + '\n')

	#evaluate NDCG
	k = 10
	bias_accum_ndcg = 0
	unbias_accum_ndcg = 0
	random_accum_ndcg = 0
	np.random.seed(2021)
	node_cnt = 0

	for node_id in range(M):
		node_edges = adj_mtx[node_id]
		pos_nodes = np.where(node_edges>0)[0]
		num_pos = len(pos_nodes)
		if num_pos == 0 or num_pos >= 100:
			continue
		neg_nodes = np.random.choice(np.where(node_edges == 0)[0], 100-num_pos, replace=False)
		all_eval_nodes = np.concatenate((pos_nodes, neg_nodes)) 
		all_eval_edges = np.zeros(100)
		all_eval_edges[:num_pos] = 1

		if eval_base:
			random_pred_edges = np.dot(X_random[node_id], X_random[all_eval_nodes].T)
			random_rank_pred_keys = np.argsort(random_pred_edges)[::-1]
			bias_pred_edges = np.dot(X_biased[node_id], X_biased[all_eval_nodes].T)
			bias_rank_pred_keys = np.argsort(bias_pred_edges)[::-1]

			random_ranked_node_edges = all_eval_edges[random_rank_pred_keys]
			bias_ranked_node_edges = all_eval_edges[bias_rank_pred_keys]
			random_ndcg = ndcg_at_k(random_ranked_node_edges, k)
			bias_ndcg = ndcg_at_k(bias_ranked_node_edges, k)
			random_accum_ndcg += random_ndcg
			bias_accum_ndcg += bias_ndcg

		unbias_pred_edges = np.dot(X_unbiased[node_id], X_unbiased[all_eval_nodes].T)		
		unbias_rank_pred_keys = np.argsort(unbias_pred_edges)[::-1]
		unbias_ranked_node_edges = all_eval_edges[unbias_rank_pred_keys]

		unbias_ndcg = ndcg_at_k(unbias_ranked_node_edges, k)
		unbias_accum_ndcg += unbias_ndcg

		node_cnt += 1

	if eval_base:
		print('bias_ndcg: ', bias_accum_ndcg/node_cnt)
		result_output.write('bias_ndcg: ' + str(bias_accum_ndcg/node_cnt) + '\n')
	print('unbias_ndcg: ', unbias_accum_ndcg/node_cnt)
	result_output.write('unbias_ndcg: ' + str(unbias_accum_ndcg/node_cnt) + '\n')
	if eval_base:
		print('random_ndcg: ', random_accum_ndcg/node_cnt)
		result_output.write('random_ndcg: ' + str(random_accum_ndcg/node_cnt) + '\n')

result_output.close()