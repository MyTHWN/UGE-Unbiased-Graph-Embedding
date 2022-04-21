"""
Make our own datasets
We store three components for each dataset
-- node_feature.csv: store node feature
-- node_label.csv: store node label
-- edge.csv: store the edges
"""

import torch
import dgl
from dgl.data import DGLDataset
from dgl import backend as F
import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import networkx as nx
import os.path

class MyDataset(DGLDataset):
    def __init__(self, data_name):
        self.data_name = data_name
        super().__init__(name='customized_dataset')

    def process(self):
        raw_folder = './raw_data'
        processed_folder = './processed_data'
        
        os.makedirs(raw_folder, exist_ok=True)
        os.makedirs(processed_folder, exist_ok=True)
        
        edge_file = '{}/{}_edge.csv'.format(processed_folder, self.data_name) 
        node_feat_file = '{}/{}_node_feature.csv'.format(processed_folder, self.data_name.split('_')[0])
        node_label_file = '{}/{}_node_label.csv'.format(processed_folder, self.data_name.split('_')[0])
        node_attribute_file = '{}/{}_node_attribute.csv'.format(processed_folder, self.data_name.split('_')[0])  # sensitive node attributes predefined to debias
        
        ### download raw data and process into unified csv format ###
        
        if not os.path.isfile(node_feat_file):
            if self.data_name.split('_')[0] == 'movielens':
                process_raw_movielens(raw_folder, processed_folder)
            elif self.data_name.split('_')[0].startswith('pokec'):
                process_raw_pokec(raw_folder, processed_folder, self.data_name.split('_')[0])
            else:
                raise FileNotFoundError('The dataset {} is not supported'.format(self.data_name.split('_')[0]))
        
        ### create dgl graph from customized data ###
        
        print ('Creating DGL graph...')
        
        # Load the data as DataFrame
        edges = pd.read_csv(edge_file, engine='python')
        node_features = pd.read_csv(node_feat_file, engine='python')
        node_labels = pd.read_csv(node_label_file, engine='python')
        node_attributes = pd.read_csv(node_attribute_file, engine='python')
        
        c = node_labels['Label'].astype('category')
        classes = dict(enumerate(c.cat.categories))
        self.num_classes = len(classes)

        # Transform from DataFrame to torch tensor
        node_features = torch.from_numpy(node_features.to_numpy()).float()
        node_labels = torch.from_numpy(node_labels['Label'].to_numpy()).long()
        edge_features = torch.from_numpy(edges['Weight'].to_numpy()).float()
        edges_src = torch.from_numpy(edges['Src'].to_numpy())
        edges_dst = torch.from_numpy(edges['Dst'].to_numpy())

        # construct DGL graph
        g = dgl.graph((edges_src, edges_dst), num_nodes=node_features.shape[0])
        g.ndata['feat'] = node_features
        g.ndata['label'] = node_labels
        g.edata['weight'] = edge_features
        
        
        # add sensitive attribute information to graph
        for l in list(node_attributes):
            g.ndata[l] = torch.from_numpy(node_attributes[l].to_numpy()).long()
                
        # rewrite the to_bidirected function to support edge weights on bidirected graph (aggregated)
        # self.graph = dgl.to_bidirected(g)
        g = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)
        g = dgl.to_simple(g, return_counts=None, copy_ndata=True, copy_edata=True)
        
        # zero in-degree nodes will lead to invalid output value
        # a common practice to avoid this is to add a self-loop
        self.graph = dgl.add_self_loop(g)

        # For node classification task, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        # ! We currently only target on link prediction task
        # ! this is a placeholder for node classification task
        n_nodes = node_features.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

        print('Finished data loading and preprocessing.')
        print('  NumNodes: {}'.format(self.graph.number_of_nodes()))
        print('  NumEdges: {}'.format(self.graph.number_of_edges()))
        print('  NumFeats: {}'.format(self.graph.ndata['feat'].shape[1]))
        # print('  NumClasses: {}'.format(self.num_classes))
        # print('  NumTrainingSamples: {}'.format(
        #     F.nonzero_1d(self.graph.ndata['train_mask']).shape[0]))
        # print('  NumValidationSamples: {}'.format(
        #     F.nonzero_1d(self.graph.ndata['val_mask']).shape[0]))
        # print('  NumTestSamples: {}'.format(
        #     F.nonzero_1d(self.graph.ndata['test_mask']).shape[0]))

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


def process_raw_movielens(raw_folder, processed_folder):
    
    ### download dataset ###
    
    print ('Downloading movielens data...')
    
    edge_file = '{}/ml-1m/ratings.dat'.format(raw_folder)
    users_file = '{}/ml-1m/users.dat'.format(raw_folder)
    items_file = '{}/ml-1m/movies.dat'.format(raw_folder)
    
    filehandle, _ = urllib.request.urlretrieve("https://files.grouplens.org/datasets/movielens/ml-1m.zip")
    zip_file_object = zipfile.ZipFile(filehandle, 'r')
    zip_file_object.extractall(raw_folder)
    
    edges = pd.read_csv(edge_file, sep='::', 
                    names=['Src', 'Dst', 'Weight', 'time'], engine='python')
    user_nodes = pd.read_csv(users_file, sep='::', 
                    names=['user', 'gender', 'age', 'occupation', 'zip'], engine='python')
    item_nodes = pd.read_csv(items_file, sep='::', 
                    names=['movie', 'title', 'genre'], encoding='latin-1', engine='python')
    
    print ('Downloaded to {}'.format(raw_folder))
    
    ### process dataset ###
    
    print ('Converting to csv format...')
    
    feature_ls = ["Action",  "Adventure",  "Animation",  "Children's",  "Comedy",
    "Crime",  "Documentary",  "Drama",  "Fantasy",  "Film-Noir",
    "Horror",  "Musical",  "Mystery",  "Romance",  "Sci-Fi",
    "Thriller",  "War",  "Western"]
    
    sensitive_attributes_predefined = ['gender', 'occupation', 'age']
    
    user_num = 6040
    item_num = 3952
    
    user_nodes['label'] = 1
    item_nodes['label'] = 0

    edges['Src'] = edges['Src'].astype(int) - 1
    edges['Dst'] = edges['Dst'].astype(int) + user_num - 1
    user_nodes['user'] = user_nodes['user'].astype(int) - 1 
    item_nodes['movie'] = item_nodes['movie'].astype(int) + user_num - 1 

    weighted_edges = edges[['Src', 'Dst', 'Weight']]
    
    user_features = {genre:np.zeros(user_num) for genre in feature_ls}
    item_features = {genre:np.zeros(item_num) for genre in feature_ls}

    for row in item_nodes.iterrows():
        idx = row[0]
        item = int(row[1]['movie']) - user_num
        genres = row[1]['genre']
        genres = genres.strip().split('|')

        for genre in genres:
            item_features[genre][item] = 1

    for row in edges.iterrows(): 
        edge = row[1]
        user = int(edge['Src'])
        item = int(edge['Dst']) - user_num

        for genre in feature_ls:
            if item_features[genre][item] == 1:
                user_features[genre][user] = 1

    user_features = pd.DataFrame(user_features)
    item_features = pd.DataFrame(item_features)

    node_features = pd.concat([user_features, item_features], ignore_index=True)
    node_labels = pd.DataFrame(
            {'Label':np.concatenate((np.ones(user_num), np.zeros(item_num)))})
    
    user_attributes = user_nodes.filter(sensitive_attributes_predefined).replace(['F', 'M'], [0, 1])
    item_attributes = pd.DataFrame(-1, index=np.arange(len(node_labels)-len(user_attributes)), columns=sensitive_attributes_predefined)
    node_attributes = pd.concat([user_attributes, item_attributes], ignore_index=True)
    
    node_attributes.to_csv('{}/movielens_node_attribute.csv'.format(processed_folder), sep=',', index=False)
    node_features.to_csv('{}/movielens_node_feature.csv'.format(processed_folder), sep=',', index=False)
    node_labels.to_csv('{}/movielens_node_label.csv'.format(processed_folder), sep=',', index=False)
    weighted_edges.to_csv('{}/movielens_edge.csv'.format(processed_folder), sep=',', index=False)
    
    print ('statistics: #user={}, #item={}, #user_feature={}, #item_feature={}'.format(len(user_nodes), len(item_nodes), len(user_features), len(item_features)))
    print ('Processed data to {}'.format(processed_folder))


def process_raw_pokec(raw_folder, processed_folder, data_name):
    
    print ('Converting to csv format...' )
    
    if data_name == 'pokec-z':
        edge_file = '{}/pokec/region_job_relationship.txt'.format(raw_folder)
        node_file = '{}/pokec/region_job.csv'.format(raw_folder)
        
    elif data_name == 'pokec-n':
        edge_file = '{}/pokec/region_job_2_relationship.txt'.format(raw_folder)
        node_file = '{}/pokec/region_job_2.csv'.format(raw_folder)

    edges = pd.read_csv(edge_file, sep='\t', names=['Src', 'Dst'], engine='python')
    nodes = pd.read_csv(node_file, sep=',', header=0, engine='python')
    
    print ('-- raw data loaded')
    
    feature_ls = ['Label', 'user_id', 'public', 
                  'completion_percentage', 'gender', 'region', 'AGE']
    
    sensitive_attributes_predefined = ['gender', 'region', 'AGE']

    node_ids = list(nodes['user_id'])
    new_ids = list(range(len(node_ids)))

    id_map = dict(zip(node_ids, new_ids))
    nodes['Label'] = nodes['public']
    node_labels = nodes.filter(['Label']) 

    edges['Weight'] = np.ones(edges.shape[0])
    edges['Src'].replace(id_map, inplace=True)
    edges['Dst'].replace(id_map, inplace=True)

    node_attributes = nodes.filter(sensitive_attributes_predefined)
    node_features = nodes.drop(columns=feature_ls)
    
    print ('-- feature and attribute filtered')

    node_attributes.to_csv('{}/{}_node_attribute.csv'.format(processed_folder, data_name), sep=',', index=False)
    node_features.to_csv('{}/{}_node_feature.csv'.format(processed_folder, data_name), sep=',', index=False)
    node_labels.to_csv('{}/{}_node_label.csv'.format(processed_folder, data_name), sep=',', index=False)
    edges.to_csv('{}/{}_edge.csv'.format(processed_folder, data_name), sep=',', index=False)
    
    print ('Processed data to {}'.format(processed_folder))











