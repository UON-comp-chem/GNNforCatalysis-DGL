#!/usr/bin/env python
# coding: utf-8
# Author Xinyu Li
import os
import sys
sys.path.append(('/home/xinyu/WSL-workspace/Repos/GNNforCatalysis-DGL'))

def test_model_schnet():
    import dgl
    import torch
    from catgnn.nmpeu import NMPEUModel
    g = dgl.DGLGraph()
    g.add_nodes(4)
    g.add_edges([0, 0, 1, 1, 1, 2, 3], [1, 0, 1, 0, 2, 3, 2])
    g.edata["distance"] = torch.tensor([1.0, 3.0, 2.0, 4.8, 2.8, 4., 6.]).reshape(-1, 1)
    g.ndata["node_type"] = torch.LongTensor([1, 2, 3, 4])
    model = NMPEUModel(embed = 'atom')
    atom = model(g)
    assert atom.shape == torch.Size([1, 1])
