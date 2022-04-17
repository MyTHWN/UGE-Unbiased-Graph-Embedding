import dgl
from dgl.nn import GraphConv    # Define a GCN model
from dgl.nn import GATConv      # Define a GAT model
from dgl.nn import SGConv       # Define a SGC model
from dgl.nn import SAGEConv     # Define a GraphSAGE model
import torch
import torch.nn as nn
import torch.nn.functional as F


######################################################################
# build a two-layer SGConv model
class SGC(nn.Module):
    def __init__(self, graph, in_dim, hidden_dim, out_dim):
        super(SGC, self).__init__()
        self.conv = SGConv(in_feats=in_dim, 
                             out_feats=out_dim, 
                             k=2)
        self.graph = graph
    
    def forward(self, in_feat):
        h = self.conv(self.graph, in_feat)
        return h

######################################################################
# build a two-layer vanilla GCN model
class GCN(nn.Module):
    def __init__(self, graph, in_dim, hidden_dim, out_dim):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats=in_dim, 
                               out_feats=hidden_dim, 
                               norm='both', weight=True, bias=True)
        self.conv2 = GraphConv(in_feats=hidden_dim, 
                               out_feats=out_dim, 
                               norm='both', weight=True, bias=True)
        self.graph = graph
    
    def forward(self, in_feat):
        h = self.conv1(self.graph, in_feat)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        h = self.conv2(self.graph, h)
        return h
    
######################################################################
# build a two-layer GAT model
class GATLayer(nn.Module):
    def __init__(self, graph, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.graph = graph
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # attention
        z2 = torch.cat([edges.src['z'],  edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, in_feat):
        z = self.fc(in_feat)
        self.graph.ndata['z'] = z
        self.graph.apply_edges(self.edge_attention)
        self.graph.update_all(self.message_func, self.reduce_func)
        return self.graph.ndata.pop('h')

class MultiHeadGATLayer(nn.Module):
    def __init__(self, graph, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(graph, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))

class GAT(nn.Module):
    def __init__(self, graph, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(graph, in_dim, hidden_dim, num_heads)
        self.layer2 = MultiHeadGATLayer(graph, hidden_dim * num_heads, out_dim, 1)

    def forward(self, in_feat):
        h = self.layer1(in_feat)
        h = F.elu(h)
        h = self.layer2(h)
        return h
    
    
######################################################################
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, graph, in_dim, hidden_dim, out_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats=in_dim, 
                              out_feats=hidden_dim, 
                              aggregator_type='mean')
        self.conv2 = SAGEConv(in_feats=hidden_dim, 
                              out_feats=out_dim, 
                              aggregator_type='mean')
        self.graph = graph
    
    def forward(self, in_feat):
        h = self.conv1(self.graph, in_feat)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        h = self.conv2(self.graph, h)
        return h

######################################################################
# build a node2vec model
class Node2vec(nn.Module):
    def __init__(self, graph, in_dim, out_dim):
        super(Node2vec, self).__init__()
        self.embed = torch.nn.Embedding(in_dim, out_dim, sparse=False)
        self.graph = graph
    
    def forward(self, in_feat):
        h = self.embed(in_feat)
        return h