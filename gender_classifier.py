
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import pickle as pk
import argparse

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size != k:
        raise ValueError('Ranking List length < k')    
    return np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(r, k):
    sort_r = sorted(r,reverse = True)
    idcg = dcg_at_k(sort_r, k)
    if not idcg:
        print('check')
        return 0.
    return dcg_at_k(r, k) / idcg

def load_user_attributes(file, M):
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

def load_rating_matrix(file, M, N):
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


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default="movielens", 
						help='attribute to debias')
	parser.add_argument('--debias', type=str, default="gender", 
                    	choices=['gender', 'age', 'occupation'],
						help='attribute to debias')
	parser.add_argument('--model', type=str, default="gat", 
						help='embedding model')
	parser.add_argument('--loss', type=str, default="gat", 
						choices=['bpr', 'entropy'],
						help='training loss')
	args = parser.parse_args()

	M = 6040 + 1
	N = 3952 + 1
	bias_attr = args.debias
	evaluate_attr = 'gender'

	rating_sparse_mtx, rating_mtx = \
				load_rating_matrix('../ml-1m/ratings.dat', M, N)
	trn_rating_sparse_mtx, trn_rating_mtx = \
				load_rating_matrix('../ml-1m/train_ratings.dat', M, N)
	tst_rating_sparse_mtx, tst_rating_mtx = \
				load_rating_matrix('../ml-1m/test_ratings.dat', M, N)

	user_attributes = load_user_attributes('../ml-1m/users.dat', M)
	# 'F' = 0, 'M' = 1
	genders = np.array([int(user_attributes[i][0]=='M') for i in range(1, M)])
	ages = np.array([int(user_attributes[i][1]) for i in range(1, M)])
	occupations = np.array([int(user_attributes[i][2]) for i in range(1, M)])

	attribute_labels = {'gender': genders, 'age': ages, 'occupation': occupations}

	rating_mtx = rating_mtx[1:]
	rating_mtx = rating_mtx[:,1:]
	print(rating_mtx.shape)
	X_random, Y_random = np.random.rand(*(M-1,16)), np.random.rand(*(N-1,16))
	biased_embedding = pk.load(open(
		'./pokec_embeddings_lu2/{}_gat_{}_001_800_embedding.bin'.format(args.dataset.split('_')[0], args.loss),'rb'))
	X_biased = biased_embedding[:M-1]
	Y_biased = biased_embedding[M-1:]

	unbiased_embedding = pk.load(open(
		'./pokec_embeddings_lu2/{}_weighting_{}_{}_{}_001_01_800_embedding.bin'.format(args.dataset, 
		args.debias, args.model, args.loss),'rb'))

	X_unbiased = unbiased_embedding[:M-1]
	Y_unbiased = unbiased_embedding[M-1:]

	for evaluate_attr in ['gender', 'age', 'occupation']:
		rating_lgreg = LogisticRegression(random_state=0, class_weight='balanced', 
			max_iter=1000).fit(rating_mtx[:5000], attribute_labels[evaluate_attr][:5000])
		bias_lgreg = LogisticRegression(random_state=0, class_weight='balanced', 
			max_iter=1000).fit(X_biased[:5000], attribute_labels[evaluate_attr][:5000])
		unbias_lgreg = LogisticRegression(random_state=0, class_weight='balanced', 
			max_iter=1000).fit(X_unbiased[:5000], attribute_labels[evaluate_attr][:5000])
		random_lgreg = LogisticRegression(random_state=0, class_weight='balanced', 
			max_iter=1000).fit(X_random[:5000], attribute_labels[evaluate_attr][:5000])

		rating_pred = rating_lgreg.predict(rating_mtx[5000:])
		base_pred = bias_lgreg.predict(X_biased[5000:])
		unbias_pred = unbias_lgreg.predict(X_unbiased[5000:])
		random_pred = random_lgreg.predict(X_random[5000:])

		print('sensitive_attribute debiased: ', bias_attr)
		print('attribute evaluated: ', evaluate_attr)
		print('raw rating micro-f1: ', f1_score(attribute_labels[evaluate_attr][5000:], 
										rating_pred, average='micro'))
		print('baseline micro-f1: ', f1_score(attribute_labels[evaluate_attr][5000:], 
										base_pred, average='micro'))
		print('debiased micro-f1: ', f1_score(attribute_labels[evaluate_attr][5000:], 
										unbias_pred, average='micro'))
		print('random embedding micro-f1: ', f1_score(attribute_labels[evaluate_attr][5000:], 
										random_pred, average='micro'))


	#evaluate NDCG
	k = 10
	bias_accum_ndcg = 0
	unbias_accum_ndcg = 0
	random_accum_ndcg = 0

	for user_id in range(1, M):
		user = user_id - 1
		user_ratings = rating_mtx[user] # (rating_mtx[user] > 0).astype(int)
		# if user < 10:
		# 	print(X_biased[user])
		# if user == 999: 
		# 	print(np.sum(user_ratings))
		random_pred_ratings = np.dot(X_random[user], Y_random.T)
		bias_pred_ratings = np.dot(X_biased[user], Y_biased.T)
		unbias_pred_ratings = np.dot(X_unbiased[user], Y_unbiased.T)

		# user_ratings = tst_rating_mtx[user]
		# random_pred_ratings = np.dot(X_random[user], Y_random.T) * (trn_rating_mtx[user] == 0)
		# bias_pred_ratings = np.dot(X_biased[user], Y_biased.T) * (trn_rating_mtx[user] == 0)
		# unbias_pred_ratings = np.dot(X_unbiased[user], Y_unbiased.T) * (trn_rating_mtx[user] == 0)

		random_rank_pred_keys = np.argsort(random_pred_ratings)[::-1]
		bias_rank_pred_keys = np.argsort(bias_pred_ratings)[::-1]
		unbias_rank_pred_keys = np.argsort(unbias_pred_ratings)[::-1]
		
		random_ranked_user_ratings = user_ratings[random_rank_pred_keys]
		bias_ranked_user_ratings = user_ratings[bias_rank_pred_keys]
		unbias_ranked_user_ratings = user_ratings[unbias_rank_pred_keys]

		random_ndcg = ndcg_at_k(random_ranked_user_ratings, k)
		bias_ndcg = ndcg_at_k(bias_ranked_user_ratings, k)
		unbias_ndcg = ndcg_at_k(unbias_ranked_user_ratings, k)

		random_accum_ndcg += random_ndcg
		bias_accum_ndcg += bias_ndcg
		unbias_accum_ndcg += unbias_ndcg

	print('bias_ndcg: ', bias_accum_ndcg/M)
	print('unbias_ndcg: ', unbias_accum_ndcg/M)
	print('random_ndcg: ', random_accum_ndcg/M)

