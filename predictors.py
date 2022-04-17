"""
Define predictor layers to serve for downstream tasks
"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class DotLinkPredictor(nn.Module):
    """
    Dot product to compute the score of link 
    The benefit of treating the pairs of nodes as a graph is that the score
    on edge can be easily computed via the ``DGLGraph.apply_edges`` method
    """

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]

class MLPLinkPredictor(nn.Module):
    """MLP to predict the score of link
    """
    
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

class MLPNodePredictor(nn.Module):
    """MLP to predict the logits for node classification
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, h):
        logits = F.softmax(self.fc(h), dim=1)
        return logits
            
