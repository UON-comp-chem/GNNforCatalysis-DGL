# -*- coding:utf-8 -*-
"""
Example dataloader of Tencent alchemy Dataset
https://github.com/tencent-alchemy/Alchemy/blob/master/dgl/Alchemy_dataset.py
"""
import os
import zipfile
import os.path as osp
import dgl
from dgl.data.utils import (download,
                            save_graphs,
                            load_graphs)
import pickle
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from .mongo import make_atoms_from_doc
from ase.db import connect
from pathlib import Path
from pymatgen.io.ase import AseAtomsAdaptor
import warnings


GP =  {
    1: [1, 1],  2: [1, 18],\
    3: [2, 1],   4: [2, 2],\
    5: [2, 13],  6: [2, 14], 7: [2, 15], 8: [2, 16], 9: [2, 17], 10: [2, 18],\
    11: [3, 1], 12: [3, 2],\
    13: [3, 13], 14: [3, 14], 15: [3, 15], 16: [3, 16], 17: [3, 17], 18: [3, 18],\
    
    19: [4, 1], 20: [4, 2],\
    21: [4, 3], 22: [4, 4], 23: [4, 5], 24: [4, 6], 25: [4, 7], 26: [4, 8], 27: [4, 9],\
    28: [4, 10], 29: [4, 11], 30: [4, 12], 31: [4, 13], 32: [4, 14], 33: [4, 15], \
    34: [4, 16], 35: [4, 17], 36: [4, 18],\
    
    37: [5, 1], 38: [5, 2], 39: [5, 3], 40: [5, 4], 41: [5, 5], 42: [5, 6], 43: [5, 7],\
    44: [5, 8], 45: [5, 9], 46: [5, 10], 47: [5, 11], 48: [5, 12], 49: [5, 13],\
    50: [5, 14], 51: [5, 15], 52: [5, 16], 53: [5, 17], 54: [5, 18], 55: [6, 1],\
    
    56: [6, 2], 57: [6, 3], 58: [6, 19], 59: [6, 20], 60: [6, 21], 61: [6, 22],\
    62: [6, 23], 63: [6, 24], 64: [6, 25], 65: [6, 26], 66: [6, 27], \
    67: [6, 28], 68: [6, 29], 69: [6, 30], 70: [6, 31], 71: [6, 32], \
    72: [6, 4], 73: [6, 5], 74: [6, 6], 75: [6, 7], 76: [6, 8], 77: [6, 9],\
    78: [6, 10], 79: [6, 11], 80: [6, 12], 81: [6, 13], 82: [6, 14], 83: [6, 15],\
    84: [6, 16], 85: [6, 17], 86: [6, 18],\
    
    87: [7, 1], 88: [7, 2],  89: [7, 3],  90: [7, 19],  91: [7, 20],\
    92: [7, 21], 93: [7, 22], 94: [7, 23], 95: [7, 24], 96: [7, 25],\
    97: [7, 26], 98: [7, 27], 99: [7, 28], 100: [7, 29], 101: [7, 30],\
    102: [7, 31], 103: [7, 32], 104: [7, 4]
    }

class Batcher:
    def __init__(self, graph=None, label=None):
        self.graph = graph
        self.label = label

def batcher():
    def batcher_dev(batch):
        graphs, labels = zip(*batch)
        batch_graphs = dgl.batch(graphs)
        labels = torch.stack(labels, 0)
        return Batcher(graph=batch_graphs, label=labels)

    return batcher_dev


class MoleculeDataset(Dataset):
    """
    Transfer ASE atoms into DGL graph
    """

    def moelcule_nodes(self, structs):
        """
        Featurization for all atoms in a molecule. The atom indices
        will be preserved.

        Args:
            structs : pymatgen.Structures
              pymatgen Structures
        Returns
            atom_feats_dict : dict
             Dictionary for atom features
        """
        atom_feats_dict = {}

        num_atoms = len(structs)
        atomic_number = structs.get_atomic_numbers()
        atom_feats_dict['pos'] = structs.positions
        atom_feats_dict['node_type'] = atomic_number
        gps = np.array([GP[an] for an in atomic_number]) #Group and Periods
        atom_feats_dict['period'] = gps[:, 0]
        atom_feats_dict['group'] = gps[:, 1]

        atom_feats_dict['pos'] = torch.FloatTensor(atom_feats_dict['pos'])
        atom_feats_dict['node_type'] = torch.LongTensor(
            atom_feats_dict['node_type'])
        atom_feats_dict['group'] = torch.LongTensor(
            atom_feats_dict['group'])
        atom_feats_dict['period'] = torch.LongTensor(
            atom_feats_dict['period'])


        return atom_feats_dict

    def molecule_to_dgl(self, structs, self_loop=False):
        """
        Read pymatgen structs and convert to dgl_graph
        Args:
            structs: pymatgen structures
            self_loop: Whetaher to add self loop
        Returns:
            g: DGLGraph
            l: related labels
        """
        g = dgl.DGLGraph()

        # add nodes
        num_atoms = len(structs)
        atom_feats = self.moelcule_nodes(structs)
        g.add_nodes(num=num_atoms, data=atom_feats)

        # add edges
        # Cancel the for loop, hope this is faster
        edge_index = [[], []]
        edge_attr = []
        pos = structs.positions.copy()
        natoms = pos.shape[0]
        xpos = pos.reshape((natoms, 1, pos.shape[1]))
        ypos =  pos.reshape((1, natoms, pos.shape[1]))
        x = np.tile(xpos, (1,natoms,  1))
        y = np.tile(ypos, (natoms, 1, 1))
        vectors= (y - x).reshape((int(natoms**2), 3))
        dis = np.linalg.norm(vectors, axis = -1)
        idxs = np.arange(natoms)
        xyidxs = np.ones((2,int(natoms**2) ))
        xyidxs[1] = np.repeat(idxs, natoms)
        xyidxs[0] = np.tile(idxs, natoms)
        edge_attr = np.ones((int(natoms**2), 4))
        edge_attr[:, 0] = dis
        edge_attr[:, 1:] = vectors
        filters = (xyidxs[1] != xyidxs[0]) & (dis < self.cutoff)
        edge_index = xyidxs[:, filters].astype(np.int)
        edge_attr = edge_attr[filters, :] 
        
        g.add_edges(edge_index[0], edge_index[1])
        bond_feats = defaultdict()
        bond_feats['distance'] = torch.FloatTensor(edge_attr[:, 0]).reshape(-1, 1)
        g.edata.update(bond_feats)
        return g

    def __init__(self, root = None, structures = [None],  targets = [None], cutoff=5., transform=None):       
        self.transform = transform
        self.cutoff = cutoff
        assert len(structures) == len(targets), "Number of structures unequal to y"
        self.structures = structures
        self.targets = targets
        self.root = root
        if self.root == None:
            self._load()
        else:
            if os.path.isdir(root) is False:
                os.mkdir(root)
            cache_path = os.path.join(root, "processed.bin")
            if os.path.isfile(cache_path):
                try:
                    graphs, labels = load_graphs(cache_path) 
                    self.graphs = graphs
                    self.labels = labels['labels']
                    print(len(self.graphs), "loded!")
                except:
                    self._load()
                    labels_dict = {"labels":self.labels}
                    save_graphs(cache_path,self.graphs, labels_dict)
                    print(len(self.graphs), "saved!")
            else:
                self._load()
                labels_dict = {"labels":self.labels}
                save_graphs(cache_path,self.graphs, labels_dict)
                print(len(self.graphs), "saved!")
                
        
    def _load(self):
        length = len(self.structures)
        self.graphs = []
        self.labels = []
        for idx, structs in enumerate(self.structures): # Need to random distribute the samples
            result = self.molecule_to_dgl(structs)
            self.graphs.append(result)
            self.labels.append(torch.FloatTensor([self.targets[idx]]))
        self.labels = torch.FloatTensor(self.labels)
        self.normalize()
        print(len(self.graphs), "processed!")

    def normalize(self, mean=None, std=None):
        labels = np.array([i.numpy() for i in self.labels])
        if mean is None:
            mean = np.mean(labels, axis=0)
        if std is None:
            std = np.std(labels, axis=0)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g, l = self.graphs[idx], self.labels[idx]
        if self.transform:
            g = self.transform(g)
        return g, l

class DGLDataset(Dataset):
    def cutoff_filter(self, edges): 
        return (edges.data['distance'] > self.cutoff).squeeze(1)

    def filter_out(self):
        graphs = []
        for idx in range(len(self.targets)):
            g = self.graphs[idx]
            e_idx = g.filter_edges(self.cutoff_filter)
            g.remove_edges(e_idx)
            graphs.append(g)
        self.graphs = graphs

    def normalize(self, mean=None, std=None):
        labels = np.array([i.numpy() for i in self.labels])
        if mean is None:
            mean = float(np.mean(labels, axis=0))
        if std is None:
            std = float(np.std(labels, axis=0))
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g, l = self.graphs[idx], self.labels[idx]
        if self.transform:
            g = self.transform(g)
        return g, l  
    
class CrystalGraphDataset(DGLDataset):
    """
    Transfer Pymatgen structs crystals into DGL graph
    """
    def __init__(self, root = None, structures = [None], targets = [None], cutoff=6., transform=None):       
        self.transform = transform
        self.cutoff = cutoff
        self.root = root
        assert len(structures) == len(targets), "Number of structures unequal to y"
        self.structures = structures
        
        self.targets = np.array(targets)
        if self.targets.ndim == 1:
            self.outdim = 1
        else:
            self.outdim = self.targets.shape[-1]
        if self.root == None:
            self._load()
        else:
            if os.path.isdir(root) is False:
                os.mkdir(root)
            cache_path = os.path.join(root, "processed.bin")
            if os.path.isfile(cache_path):
                try:
                    graphs, labels = load_graphs(cache_path) 
                    self.graphs = graphs
                    self.labels = labels['labels']
                    print(len(self.graphs), "loded!")
                except:
                    self._load()
                    labels_dict = {"labels":self.labels}
                    save_graphs(cache_path,self.graphs, labels_dict)
                    print(len(self.graphs), "saved!")
            else:
                self._load()
                labels_dict = {"labels":self.labels}
                save_graphs(cache_path, self.graphs, labels_dict)
                print(len(self.graphs), "saved!")
        self.filter_out()
        self.normalize()

    def crystal_nodes(self, structs):
        """
        Featurization for all atoms in a molecule. The atom indices
        will be preserved.

        Args:
            structs : pymatgen.Structures
              pymatgen Structures
        Returns
            atom_feats_dict : dict
             Dictionary for atom features
        """
        atom_feats_dict = defaultdict(list)
        
        num_atoms = len(structs)
        for idx, atom in enumerate(structs):
            symbol = atom.specie
            atom_type = symbol.number
            period, group = GP[atom_type]
            atom_feats_dict['pos'].append(torch.FloatTensor([atom.x, atom.y, atom.z]))
            atom_feats_dict['node_type'].append(atom_type)
            atom_feats_dict['period'].append(period)
            atom_feats_dict['group'].append(group)

        atom_feats_dict['pos'] = torch.stack(atom_feats_dict['pos'], dim=0)
        atom_feats_dict['node_type'] = torch.LongTensor(
            atom_feats_dict['node_type'])
        atom_feats_dict['group'] = torch.LongTensor(
            atom_feats_dict['group'])
        atom_feats_dict['period'] = torch.LongTensor(
            atom_feats_dict['period'])

        return atom_feats_dict

    def structs_to_dgl(self, structs, self_loop=False):
        """
        Read pymatgen structs and convert to dgl_graph
        Args:
            structs: pymatgen structures
            self_loop: Whetaher to add self loop
        Returns:
            g: DGLGraph
            l: related labels
        """

        g = dgl.DGLGraph()

        # add nodes
        num_atoms = len(structs)
        atom_feats = self.crystal_nodes(structs)
        g.add_nodes(num=num_atoms, data=atom_feats)

        # add edges
        # Different with molecules, it is multi-edges used here
        # Hacked from https://github.com/txie-93/cgcnn/blob/master/cgcnn/data.py
        # 9. is the maximum cutoff
        all_nbrs = structs.get_all_neighbors(9., include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea, nbr_pos = [], [], []
        for nbr in all_nbrs:
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) )
            nbr_fea.append(list(map(lambda x: x[1], nbr)) )
            nbr_pos.append(list(map(lambda x: x[0].coords,
                            nbr)))

        src, dst = [], []
        distances = []
        vectors = []
        for idx, nbrs in enumerate(nbr_fea_idx):
            for idy, nbr in enumerate(nbrs):
                src.append(nbr)
                dst.append(idx)
                distances.append(nbr_fea[idx][idy])
                vectors.append(nbr_pos[idx][idy] -structs[idx].coords)
        
        g.add_edges(src, dst)
        bond_feats = defaultdict()
        bond_feats['distance'] = torch.FloatTensor(distances).reshape(-1, 1)
        bond_feats['vector'] = torch.FloatTensor(vectors).reshape(-1, 3)
        g.edata.update(bond_feats)
        g = dgl.add_self_loop(g)
        return g
        
    def _load(self):
        length = len(self.structures)
        self.graphs, self.labels = [], []
        for idx, structs in enumerate(self.structures): # Need to random distribute the samples
            result = self.structs_to_dgl(structs)
            self.graphs.append(result)
            self.labels.append(torch.FloatTensor([self.targets[idx]]))
        self.labels = torch.FloatTensor(self.labels).view(-1, self.outdim)
        print(len(self.graphs), "loaded!")

    
class CrystalGraphDatasetLS(CrystalGraphDataset):
    """
    Transfer Pymatgen structs crystals into DGL graph with Labelled Sites
    """
    def __init__(self, root = None, structures = [None], labelsites = [None], targets = [None], cutoff=6., transform=None): 
        self.labelsites = labelsites
        assert len(structures) == len(labelsites), "Number of structures unequal to labelsites"
        super(CrystalGraphDatasetLS, self).__init__(root = root, 
                                                structures = structures, 
                                                targets = targets, 
                                                cutoff = cutoff,
                                                transform = transform)

    def crystal_nodes(self, structs, ls):
        """
        Featurization for all atoms in a molecule. The atom indices
        will be preserved.

        Args:
            structs : pymatgen.Structures
              pymatgen Structures
        Returns
            atom_feats_dict : dict
             Dictionary for atom features
        """
        atom_feats_dict = defaultdict(list)
        
        num_atoms = len(structs)
        for idx, atom in enumerate(structs):
            symbol = atom.specie
            atom_type = symbol.number
            period, group = GP[atom_type]
            atom_feats_dict['pos'].append(torch.FloatTensor([atom.x, atom.y, atom.z]))
            atom_feats_dict['node_type'].append(atom_type)
            atom_feats_dict['period'].append(period)
            atom_feats_dict['group'].append(group)
            atom_feats_dict['ls'].append(ls[idx])

        atom_feats_dict['pos'] = torch.stack(atom_feats_dict['pos'], dim=0)
        atom_feats_dict['node_type'] = torch.LongTensor(
            atom_feats_dict['node_type'])
        atom_feats_dict['group'] = torch.LongTensor(
            atom_feats_dict['group'])
        atom_feats_dict['period'] = torch.LongTensor(
            atom_feats_dict['period'])
        atom_feats_dict['ls'] = torch.LongTensor(
            atom_feats_dict['ls'])

        return atom_feats_dict
    
    def _load(self):
        length = len(self.structures)
        self.graphs, self.labels = [], []
        for idx, structs in enumerate(self.structures): # Need to random distribute the samples
            result = self.structs_to_dgl(structs, self.labelsites[idx])
            self.graphs.append(result)
            self.labels.append(torch.FloatTensor([self.targets[idx]]))
        self.labels = torch.FloatTensor(self.labels).view(-1, self.outdim)
        print(len(self.graphs), "loaded!")
        
    def structs_to_dgl(self, structs, ls, self_loop=False):
        """
        Read pymatgen structs and convert to dgl_graph
        Args:
            structs: pymatgen structures
            self_loop: Whetaher to add self loop
        Returns:
            g: DGLGraph
            l: related labels
        """

        g = dgl.DGLGraph()

        # add nodes
        num_atoms = len(structs)
        atom_feats = self.crystal_nodes(structs, ls)
        g.add_nodes(num=num_atoms, data=atom_feats)

        # add edges
        # Different with molecules, it is multi-edges used here
        # Hacked from https://github.com/txie-93/cgcnn/blob/master/cgcnn/data.py
        all_nbrs = structs.get_all_neighbors(9., include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea, nbr_pos = [], [], []
        for nbr in all_nbrs:
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) )
            nbr_fea.append(list(map(lambda x: x[1], nbr)) )
            nbr_pos.append(list(map(lambda x: x[0].coords,
                            nbr)))            
        src, dst = [], []
        distances = []
        vectors = []
        for idx, nbrs in enumerate(nbr_fea_idx):
            for idy, nbr in enumerate(nbrs):
                src.append(nbr)
                dst.append(idx)
                distances.append(nbr_fea[idx][idy])
                vectors.append(nbr_pos[idx][idy] -structs[idx].coords)
        
        g.add_edges(src, dst)
        bond_feats = defaultdict()
        bond_feats['distance'] = torch.FloatTensor(distances).reshape(-1, 1)
        bond_feats['vector'] = torch.FloatTensor(vectors).reshape(-1, 3)
        g.edata.update(bond_feats)
        g = dgl.add_self_loop(g)
        return g

        
