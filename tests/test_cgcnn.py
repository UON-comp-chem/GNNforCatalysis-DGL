#!/usr/bin/env python
# coding: utf-8
# Author Xinyu Li
# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
import sys
sys.path.append(('/home/xinyu/WSL-workspace/Repos/GNNforCatalysis-DGL'))

def test_model_cgcnn1():
    import dgl
    import torch
    from catgnn.cgcnn import CGCNN
    g = dgl.DGLGraph()
    g.add_nodes(4)
    g.add_edges([0, 0, 1, 1, 1, 2, 3], [1, 0, 1, 0, 2, 3, 2])
    g.edata["distance"] = torch.tensor([1.0, 3.0, 2.0, 4.8, 2.8, 4., 6.]).reshape(-1, 1)
    g.ndata["node_type"] = torch.LongTensor([1, 2, 3, 4])
    model = CGCNN(embed = 'atom')
    atom = model(g)
    assert atom.shape == torch.Size([1, 1])

def test_model_cgcnn2():
    import dgl
    import torch
    from catgnn.cgcnn import CGCNN
    g = dgl.DGLGraph()
    g.add_nodes(4)
    g.add_edges([0, 0, 1, 1, 1, 2, 3], [1, 0, 1, 0, 2, 3, 2])
    g.edata["distance"] = torch.tensor([1.0, 3.0, 2.0, 4.8, 2.8, 4., 6.]).reshape(-1, 1)
    g.ndata["node_type"] = torch.LongTensor([1, 2, 3, 4])
    model = CGCNN(embed = 'atom', norm = True)
    model.set_mean_std(1.0, 1.0)
    atom = model(g)
    assert atom.shape == torch.Size([1, 1])
