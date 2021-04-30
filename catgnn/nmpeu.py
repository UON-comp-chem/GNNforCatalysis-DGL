#!/usr/bin/env python
# coding: utf-8
# Author Xinyu Li

"""
The Neural Message Passing with Edge Updates model
Ref:
https://arxiv.org/abs/1806.03146
"""


import dgl
import torch
import torch.nn as nn
import numpy as np
import dgl.function as fn
from torch.nn import Softplus
from dgl.nn.pytorch.conv.cfconv import CFConv, ShiftedSoftplus
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.glob import SumPooling, AvgPooling
import numpy as np
from .layers import (GPLSEmbedding, GPEmbedding, AtomEmbedding, 
                     FakeAtomEmbedding,RBFLayer)
                    
PI = float(np.pi)
    
class EdgeUpdate(nn.Module):
    def __init__(self, rbf_dim = 64, dim=64, act="sp"):
        super().__init__()
        self.activation = act
        self.project_edge = nn.Sequential(
                    nn.Linear(int(dim*2) + rbf_dim, int(dim*2)),
                    self.activation,
                    nn.Linear(int(dim*2), rbf_dim))
        
    def update_edge(self, edges):
        rbf = torch.cat((edges.src['h'], edges.dst['h'], edges.data['rbf']), -1)
        return {'edge_h': rbf}
        
    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            g.ndata['h'] = node_feats
            g.edata['rbf'] = edge_feats
            g.apply_edges(self.update_edge)
            edge_feats = self.project_edge(g.edata['edge_h'])
        return edge_feats
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.project_edge[0].weight)
        self.project_edge[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.project_edge[2].weight)
        self.project_edge[2].bias.data.fill_(0)
            
        
class EUCFConv(nn.Module):
    """
    The continuous-filter convolution layer in SchNet with Edge Updates
    """

    def __init__(self, rbf_dim = 64, dim=64, act="sp"):
        """
        Args:
            rbf_dim: the dimsion of the RBF layer
            dim: the dimension of linear layers
            act: activation function (default shifted softplus)
        """
        super().__init__()
        self._dim = dim
        self.activation = act
        self.project_node1 = nn.Sequential(
                            nn.Linear(self._dim, self._dim),
                            self.activation)
        self.project_edge = nn.Sequential(
                            nn.Linear(rbf_dim, self._dim),
                            self.activation, 
                            nn.Linear(self._dim, self._dim))
        self.project_node2 = nn.Sequential(
                            nn.Linear(self._dim, self._dim),
                            self.activation, 
                            nn.Linear(self._dim, self._dim))        

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            g.ndata['hv'] = self.project_node1(node_feats)
            g.edata['he'] = self.project_edge(edge_feats)
            g.update_all(message_func=fn.copy_edge('he', 'm'),
                     reduce_func=fn.sum('m', 'h'))
            new_node = self.project_node2(g.ndata['h'])
        return new_node
    
    def reset_parameters(self):
        """Reinitialize model parameters."""
        torch.nn.init.xavier_uniform_(self.project_node1[0].weight)
        self.project_node1[0].bias.data.fill_(0)
        
        torch.nn.init.xavier_uniform_(self.project_edge[0].weight)
        self.project_edge[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.project_edge[2].weight)
        self.project_edge[2].bias.data.fill_(0)
        
        torch.nn.init.xavier_uniform_(self.project_node2[0].weight)
        self.project_node2[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.project_node2[2].weight)
        self.project_node2[2].bias.data.fill_(0)

class NMPEUInteraction(nn.Module):
    """
    The interaction layer in the NMPEU model.
    """

    def __init__(self, rbf_dim, dim, act):
        super().__init__()
        self._node_dim = dim
        self.activation = act
        self.edge_update = EdgeUpdate(rbf_dim = rbf_dim, 
                                      dim=dim, 
                                      act=self.activation)
        self.interaction = EUCFConv(rbf_dim = rbf_dim,
                                 dim = dim,
                                 act = self.activation)


    def forward(self, g, node_feats, edge_feats):
        edge_feats = self.edge_update(g, node_feats, edge_feats)
        new_node  = self.interaction(g, node_feats, edge_feats)
        node_feats += new_node
        return node_feats, edge_feats
    
    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.edge_update.reset_parameters()
        self.interaction.reset_parameters()
    

class NMPEUModel(nn.Module):
    """
    NMPEU Model
    """

    def __init__(self,
                 embed = "atom",
                 dim=64,
                 cutoff=5.,
                 output_dim=1,
                 num_gaussians = 64, 
                 n_conv=3,
                 act = "ssp",
                 aggregation_mode="avg",
                 norm=False):
        """
        Args:
            dim: dimension of features
            output_dim: dimension of prediction
            cutoff: radius cutoff
            width: width in the RBF function
            n_conv: number of interaction layers
            norm: normalization
        """
        super().__init__()
        self.name = "NMPEUModel"
        self._dim = dim
        self.cutoff = cutoff
        self.num_gaussians= num_gaussians
        self.n_conv = n_conv
        self.norm = norm
        self.activation = ShiftedSoftplus()
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
            [NMPEUInteraction(self.rbf_layer._fan_out, dim, act = self.activation) for i in range(n_conv)])

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

        # Update the edge
        for idx in range(self.n_conv):
            node_feats, edge_feats = self.conv_layers[idx](g, node_feats, edge_feats)

        atom = self.atom_dense_layer1(node_feats)
        atom = self.activation(atom)
        res = self.atom_dense_layer2(atom)
        if self.norm:
            res = res * self.std + self.mean
        res = self.readout(g, res)
        
        return res
    
    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.embedding_layer.reset_parameters()
        for conv in self.conv_layers:
            conv.reset_parameters()
        for pool in self.pool_layers:
            pool.reset_parameters()
            
        torch.nn.init.xavier_uniform_(self.atom_dense_layer1.weight)
        self.atom_dense_layer1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.atom_dense_layer2.weight)
        self.atom_dense_layer2.bias.data.fill_(0)      

