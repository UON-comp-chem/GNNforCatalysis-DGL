#!/usr/bin/env python
# coding: utf-8
# Author Xinyu Li

import dgl
import torch
import torch.nn as nn
from .layers import (AtomEmbedding, SchInteraction, RBFLayer,
                     GPLSEmbedding, GPEmbedding)
from dgl.nn.pytorch.conv.cfconv import ShiftedSoftplus
from dgl.nn.pytorch.glob import SumPooling, AvgPooling

class SchNet(nn.Module):
    """
    SchNet Model from:
        Schütt, Kristof, et al.
        SchNet: A continuous-filter convolutional neural network
        for modeling quantum interactions. (NIPS'2017)
    """

    def __init__(self,
                 embed = "gpls",
                 dim=64,
                 hidden_dim = 64, 
                 num_gaussians=64,
                 cutoff=5.0,
                 output_dim=1,
                 n_conv=3,
                 act = ShiftedSoftplus(),
                 aggregation_mode = 'avg',
                 norm=False):
        """
        Args:
            embed: Group and Period embeding to atomic number
                    Embedding
            dim: dimension of features
            output_dim: dimension of prediction
            cutoff: radius cutoff
            num_gaussians: dimension in the RBF function
            n_conv: number of interaction layers
            norm: normalization
        """
        super().__init__()
        self.name = "SchNet"
        self._dim = dim
        self.cutoff = cutoff
        self.n_conv = n_conv
        self.norm = norm
        self.output_dim = output_dim
        self.aggregation_mode = aggregation_mode
        
        if act == None:
            self.activation = ShiftedSoftplus()
        else:
            self.activation = act

            
        assert embed in ['gpls', 'atom', 'gp'], \
            "Expect mode to be 'gpls' or 'atom' or 'gp', got {}".format(embed)
        if embed == "gpls":
            self.embedding_layer = GPLSEmbedding(dim)
        elif embed == "atom":
            self.embedding_layer = AtomEmbedding(dim)
        elif embed == "gp":
            self.embedding_layer = GPEmbedding(dim)
            
        self.rbf_layer = RBFLayer(0, cutoff, num_gaussians)
        self.conv_layers = nn.ModuleList(
            [SchInteraction(self.rbf_layer._fan_out, dim) for i in range(n_conv)])
        self.atom_dense_layer1 = nn.Linear(dim, int(dim/2))
        self.atom_dense_layer2 = nn.Linear(int(dim/2), output_dim)
        if self.aggregation_mode == 'sum':
            self.readout = SumPooling()
        elif self.aggregation_mode == "avg":
            self.readout = AvgPooling()

    def set_mean_std(self, mean, std):
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))

    def forward(self, g):
        """g is the DGL.graph"""

        node_feats = self.embedding_layer(g)
        edge_feats = self.rbf_layer(g.edata['distance'])
        
        for idx in range(self.n_conv):
            node_feats = self.conv_layers[idx](g,node_feats, edge_feats)
        
        atom = self.atom_dense_layer1(node_feats)
        atom = self.activation(atom)
        res = self.atom_dense_layer2(atom)
        
        if self.norm:
            res = res * self.std + self.mean
        
        res = self.readout(g, res)
        return res
  
    
