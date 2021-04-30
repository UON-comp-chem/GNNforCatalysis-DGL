#!/usr/bin/env python
# coding: utf-8
# Author Xinyu Li

"""
Reference: https://github.com/txie-93/cgcnn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv, WeightAndSum
import dgl.function as fn
from dgl.nn.pytorch.glob import  SumPooling, AvgPooling
from .layers import (AtomEmbedding, GPEmbedding,
                     GPLSEmbedding,FakeAtomEmbedding,  RBFLayer)


__all__ = ['CGCNN']

class CGCNNConv(nn.Module):
    """
    The convolution layer in CGCNN.
    """
    def __init__(self, node_in_fea, edge_in_fea):
        super(CGCNNConv, self).__init__()
        self.node_in_fea = node_in_fea
        self.edge_in_fea = edge_in_fea
        self.fc_full1 = nn.Linear(2*self.node_in_fea+self.edge_in_fea,
                                 self.node_in_fea)
        self.fc_full2 = nn.Linear(2*self.node_in_fea+self.edge_in_fea,
                                 self.node_in_fea)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(self.node_in_fea)
        self.bn2 = nn.BatchNorm1d(self.node_in_fea)
        self.bn3 = nn.BatchNorm1d(self.node_in_fea)
        self.softplus2 = nn.Softplus()
        
    def update_edge(self, edges):
        total_edge = torch.cat((edges.src['node'], edges.dst['node'], edges.data['rbf']), -1)

        nbr_filter = self.fc_full1(total_edge)
        nbr_filter = self.bn1(nbr_filter)
        nbr_filter = self.sigmoid(nbr_filter)
        
        nbr_core = self.fc_full2(total_edge)
        nbr_core = self.bn2(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        
        nbr_ewm = nbr_filter * nbr_core # elements-wise multiplication
        return {"h": nbr_ewm}

    def forward(self, g, node_feats, edge_feats):
        g.local_var()
        g.ndata['node'] = node_feats
        g.edata['rbf'] = edge_feats
        g.apply_edges(self.update_edge)
        g.update_all(message_func=fn.copy_e('h', 'neighbor_info'),
                     reduce_func=fn.sum('neighbor_info', 'new_node'))
        nnode = self.bn3(g.ndata["new_node"])
        node_feats += nnode
        
        return g, node_feats, edge_feats

class CGCNN(nn.Module):
    """
    CGCNN Model from:
        Tian, Xie et al.
        Crystal Graph Convolutional Neural Networks for an Accurate and 
        Interpretable Prediction of Material Properties (PRB)
    """
    def __init__(self, 
                 embed = "gpls",
                 dim = 64, 
                 hidden_dim = 128, 
                 output_dim=1,
                 n_conv = 3, 
                 cutoff = 12., 
                 num_gaussians = 64, 
                 aggregation_mode = 'avg',
                 norm=False):
        super(CGCNN, self).__init__()
        self.name = "CGCNN"
        self.dim = dim
        self._dim = hidden_dim
        self.cutoff = cutoff
        self.n_conv = n_conv
        self.norm = norm
        self.num_gaussians = num_gaussians
        self.activation = nn.Softplus()
        self.aggregation_mode = aggregation_mode

        assert embed in ["atom", "gp", "gpls", "fakeatom"]
        if embed == "gpls":
            self.embedding_layer = GPLSEmbedding(dim)
        elif embed == "gp":
            self.embedding_layer = GPEmbedding(dim)
        elif embed == "atom":
            self.embedding_layer = AtomEmbedding(dim)
        elif embed == "fakeatom":
            self.embedding_layer = FakeAtomEmbedding(dim)
            
        self.rbf_layer = RBFLayer(0, cutoff, num_gaussians)
        self.conv_layers = nn.ModuleList(
            [CGCNNConv(self.dim, self.rbf_layer._fan_out) for i in range(n_conv)])
        
        assert aggregation_mode in ['sum', 'avg'], \
            "Expect mode to be 'sum' or 'avg', got {}".format(aggregation_mode )
        if self.aggregation_mode == 'sum':
            self.readout = SumPooling()
        elif self.aggregation_mode == "avg":
            self.readout = AvgPooling()
        
        self.conv_to_fc = nn.Linear(dim, hidden_dim)
        self.conv_to_fc_softplus = nn.Softplus()
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def set_mean_std(self, mean, std):
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))

    def forward(self, g):
        """g is the DGL.graph"""
        node_feats = self.embedding_layer(g)
        edge_feats = self.rbf_layer(g.edata['distance'])
        
        # Update the edge
        for idx in range(self.n_conv):
            g, node_feats, edge_feats = self.conv_layers[idx]( g, node_feats, edge_feats)
        
        crys_fea = self.readout(g, node_feats)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        res = self.fc_out(crys_fea)
        if self.norm:
            res = res * self.std + self.mean
        return res
    
