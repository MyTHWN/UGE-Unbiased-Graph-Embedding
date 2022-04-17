"""
Make our own datasets
We store three components for each dataset
-- node_feature.csv: store node feature
-- node_label.csv: store node label
-- edge.csv: store the edges
"""

import dgl
from dgl.data import DGLDataset
import torch
from dgl import backend as F
import os
import urllib.request
import pandas as pd
import numpy as np
import networkx as nx

class MyDataset(DGLDataset):
    def __init__(self, data_name):
        self.data_name = data_name
        super().__init__(name='customized_dataset')

    def process(self):
        # Load the data as DataFrame
        node_features = pd.read_csv('./mydata/{}_node_feature.csv'.format(self.data_name.split('_')[0]))
        node_labels = pd.read_csv('./mydata/{}_node_label.csv'.format(self.data_name.split('_')[0]))
        edges = pd.read_csv('./mydata/{}_edge.csv'.format(self.data_name))
        
        node_attributes = None
        if os.path.isfile('./mydata/{}_node_attribute.csv'.format(self.data_name.split('_')[0])):
            node_attributes = pd.read_csv('./mydata/{}_node_attribute.csv'.format(self.data_name.split('_')[0]))

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
        # !!! note to turn a directed graph into a undirected one
        # otherwise the graph embedding performance might be compromised
        g = dgl.graph((edges_src, edges_dst), num_nodes=node_features.shape[0])
        g.ndata['feat'] = node_features
        g.ndata['label'] = node_labels
        g.edata['weight'] = edge_features
        
        nx_g = dgl.to_networkx(g)
        print(nx.classes.function.density(nx_g))
        print(nx_g.number_of_nodes())
        print(nx_g.number_of_edges())
        
        if node_attributes is not None:
            for l in list(node_attributes):
                g.ndata[l] = torch.from_numpy(node_attributes[l].to_numpy()).long()
                
        # rewrite the to_bidirected function to support edge weights on bidirected graph (aggregated)
        # self.graph = dgl.to_bidirected(g)
        g = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)
        g = dgl.to_simple(g, return_counts=None, copy_ndata=True, copy_edata=True)
        
        # zero in-degree nodes will lead to invalid output value
        # a common practice to avoid this is to add a self-loop
        self.graph = dgl.add_self_loop(g)
        # self.graph = g

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
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
        print('  NumClasses: {}'.format(self.num_classes))
        print('  NumTrainingSamples: {}'.format(
            F.nonzero_1d(self.graph.ndata['train_mask']).shape[0]))
        print('  NumValidationSamples: {}'.format(
            F.nonzero_1d(self.graph.ndata['val_mask']).shape[0]))
        print('  NumTestSamples: {}'.format(
            F.nonzero_1d(self.graph.ndata['test_mask']).shape[0]))

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


def process_raw_karate():
    os.makedirs('./tmp', exist_ok=True)
    os.makedirs('./mydata', exist_ok=True)

    edge_tmp_file = './tmp/interactions.csv'
    node_tmp_file = './tmp/members.csv'

    urllib.request.urlretrieve(
        'https://data.dgl.ai/tutorial/dataset/members.csv', node_tmp_file)
    urllib.request.urlretrieve(
        'https://data.dgl.ai/tutorial/dataset/interactions.csv', edge_tmp_file)

    edges = pd.read_csv(edge_tmp_file)
    nodes = pd.read_csv(node_tmp_file)
    nodes['Label'] = nodes['Club'].astype('category').cat.codes

    node_feature = pd.DataFrame(nodes['Age'])
    node_label = nodes[['Label', 'Club']]
    
    node_feature.to_csv('./mydata/karate_node_feature.csv', sep=',')
    node_label.to_csv('./mydata/karate_node_label.csv', sep=',')
    edges.to_csv('./mydata/karate_edge.csv', sep=',')


# TODO: Define the other datasets as you like
def process_raw_movielens():
    genre_ls = ["Action",  "Adventure",  "Animation",  "Children's",  "Comedy",
    "Crime",  "Documentary",  "Drama",  "Fantasy",  "Film-Noir",
    "Horror",  "Musical",  "Mystery",  "Romance",  "Sci-Fi",
    "Thriller",  "War",  "Western"]

    os.makedirs('./mydata', exist_ok=True)

    edge_file = '../ml-1m/ratings.dat'
    users_file = '../ml-1m/users.dat'
    items_file = '../ml-1m/movies.dat'

    edges = pd.read_csv(edge_file, sep='::', 
                    names=['Src', 'Dst', 'Weight', 'time'])
    user_nodes = pd.read_csv(users_file, sep='::', 
                    names=['user', 'gender', 'age', 'occupation', 'zip'])
    item_nodes = pd.read_csv(items_file, sep='::', 
                    names=['movie', 'title', 'genre'], encoding='latin-1')

    user_nodes['label'] = 1
    item_nodes['label'] = 0

    edges['Src'] = edges['Src'].astype(int) - 1
    edges['Dst'] = edges['Dst'].astype(int) + 6040 - 1
    user_nodes['user'] = user_nodes['user'].astype(int) - 1 
    item_nodes['movie'] = item_nodes['movie'].astype(int) + 6040 - 1 

    weighted_edges = edges[['Src', 'Dst', 'Weight']]
    
    user_features = {genre:np.zeros(6040) for genre in genre_ls}
    item_features = {genre:np.zeros(3952) for genre in genre_ls}

    for row in item_nodes.iterrows():
        idx = row[0]
        item = int(row[1]['movie']) - 6040
        genres = row[1]['genre']
        genres = genres.strip().split('|')

        for genre in genres:
            item_features[genre][item] = 1

    for row in edges.iterrows(): 
        edge = row[1]
        user = int(edge['Src'])
        item = int(edge['Dst']) - 6040

        for genre in genre_ls:
            if item_features[genre][item] == 1:
                user_features[genre][user] = 1

    user_features = pd.DataFrame(user_features)
    item_features = pd.DataFrame(item_features)

    node_features = pd.concat([user_features, item_features], ignore_index=True)
    node_labels = pd.DataFrame(
            {'Label':np.concatenate((np.ones(6040), np.zeros(3952)))})
    
    user_attributes = user_nodes.filter(['gender', 'occupation', 'age']).replace(['F', 'M'], [0, 1])
    item_attributes = pd.DataFrame(-1, index=np.arange(len(node_labels)-len(user_attributes)), columns=['gender', 'occupation', 'age'])
    node_attributes = pd.concat([user_attributes, item_attributes], ignore_index=True)
    
    print(len(user_nodes), len(item_nodes), len(user_features), len(item_features))
    
    node_attributes.to_csv('./mydata/movielens_node_attribute.csv', sep=',', index=False)
    node_features.to_csv('./mydata/movielens_node_feature.csv', sep=',', index=False)
    node_labels.to_csv('./mydata/movielens_node_label.csv', sep=',', index=False)
    weighted_edges.to_csv('./mydata/movielens_edge.csv', sep=',', index=False)


def process_raw_pokec():

    os.makedirs('./mydata', exist_ok=True)

    edge_file = '../pokec/region_job_2_relationship.txt'
    node_file = '../pokec/region_job_2.csv'

    edges = pd.read_csv(edge_file, sep='\t', names=['Src', 'Dst'])
    nodes = pd.read_csv(node_file, sep=',', header=0)

    node_ids = list(nodes['user_id'])
    new_ids = list(range(len(node_ids)))

    id_map = dict(zip(node_ids, new_ids))
    nodes['Label'] = nodes['public']
    node_labels = nodes.filter(['Label']) 

    edges['Weight'] = np.ones(edges.shape[0])
    edges['Src'].replace(id_map, inplace=True)
    edges['Dst'].replace(id_map, inplace=True)

    node_attributes = nodes.filter(['gender', 'region', 'AGE'])
    node_features = nodes.drop(columns=['Label', 'user_id', 'public', 
        'completion_percentage', 'gender', 'region', 'AGE'])

    node_attributes.to_csv('./mydata/pokec2_node_attribute.csv', sep=',', index=False)
    node_features.to_csv('./mydata/pokec2_node_feature.csv', sep=',', index=False)
    node_labels.to_csv('./mydata/pokec2_node_label.csv', sep=',', index=False)
    edges.to_csv('./mydata/pokec2_edge.csv', sep=',', index=False)
    

# First process data into the unified csv format
# process_raw_karate()
process_raw_movielens()
# process_raw_pokec()










