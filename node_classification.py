"""
Node Classification Task: predicting labels of arbitrary nodes in a graph.
===========================================
-  Model: DGL-based graphsage and gat encoder (and many more)
-  Loss: cross entropy. You can modify the loss as you want
-  Metric: Accuracy
"""

from dgl import DGLGraph
import dgl.data 
import numpy as np
import time
import torch
import torch.nn.functional as F
import pandas as pd

import argparse
import os
import pickle as pkl

from models import GAT, GraphSAGE
from utils import * 
from predictors import MLPNodePredictor


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--device', type=int, default=0, help='cuda')
parser.add_argument('--model', type=str, default='sage', choices=['sage', 'gat'], help='model variant')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dim1', type=int, default=64, help='Number of first layer hidden units.')
parser.add_argument('--dim2', type=int, default=16, help='Number of second layer hidden units.')
parser.add_argument('--dataset', type=str, default='cora', help='dataset')
parser.add_argument('--out_dir', type=str, default='./pokec_embeddings_lu', help='output embedding foler')

args = parser.parse_args()

######################################################################
# Set up device and fix random seed
print ('==== Environment ====')
device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1) # limit cpu use
print ('  pytorch version: ', torch.__version__)
print ('  device: ', device)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

######################################################################
# Load and construct data for node classification task
print ('==== Dataset ====')
graph, features, labels, train_mask, valid_mask, test_mask = load_node_classification_data(args.dataset)

# node_attributes = pd.read_csv('./mydata/{}_node_attribute.csv'.format(args.dataset))
# node_attributes = torch.from_numpy(node_attributes.to_numpy()).type(torch.int64)
# #0: gender; 1: region; 2: age
# labels = node_attributes[:,2]

n_features = features.shape[1]
n_labels = int(labels.max().item() + 1)


# Initialize embedding model
if args.model == 'sage':
    model = GraphSAGE(graph,
                      in_dim=n_features, 
                      hidden_dim=args.dim1, 
                      out_dim=n_labels)
else:
    model = GAT(graph,
                in_dim=n_features,
                hidden_dim=args.dim1//8,
                out_dim=n_labels,
                num_heads=8)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

######################################################################
print ('==== Training ====')
# Training loop
dur = []
cur = time.time()
for e in range(args.epochs):
    model.train()
    # forward propagation using all nodes
    logits = model(features)

    # compute loss
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])

    # compute validation accuracy
    model.eval()
    with torch.no_grad():
        # calculate accuracy
        train_acc, train_recall, train_f1 = evaluate_acc(logits, labels, train_mask)
        valid_acc, valid_recall, valid_f1 = evaluate_acc(logits, labels, valid_mask)
        test_acc, test_recall, test_f1 = evaluate_acc(logits, labels, test_mask)

    # backward propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    dur.append(time.time() - cur)
    cur = time.time()
    
    if e % 10 == 0:
        log = "Epoch {:05d} | Loss {:.4f} | Train acc {:.4f} recall {:.4f} f1 {:.4f} | Valid acc {:.4f} recall {:.4f} f1 {:.4f} | Test acc {:.4f} recall {:.4f} f1 {:.4f} | Time {:.4f}"
        print(log.format(e, loss.item(), train_acc, train_recall, train_f1, valid_acc, valid_recall, valid_f1, test_acc, test_recall, test_f1, dur[-1]))

######################################################################
# Save embedding
embeddings = h.detach().numpy()
print ('==== saving the learned embeddings ====')
print ('Shape: {} | Type: {}'.format(embeddings.shape, type(embeddings)))
path = '{}/{}_{}_{}_{}_embedding_supervised.bin'.format(args.out_dir, args.dataset, args.model, args.lr, args.epochs)
os.makedirs(args.out_dir, exist_ok=True)
with open(path, "wb") as output_file:
    pkl.dump(embeddings, output_file)
print ('==== saved to {} ===='.format(path))