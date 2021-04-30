import torch
import torch.nn.functional as F
import dgl
import torch.nn as nn

class AtomEmbedding(nn.Module):
    """
    Convert the atom(node) list to atom embeddings.
    The atom with the same element share the same initial embeddding.
    """

    def __init__(self, dim=128, type_num=100, pre_train=None):
        """
        Randomly init the element embeddings.
        Args:
            dim: the dim of embeddings
            type_num: the largest atomic number of atoms in the dataset
            pre_train: the pre_trained embeddings
        """
        super().__init__()
        self._dim = dim
        self._type_num = type_num
        if pre_train is not None:
            self.embedding = nn.Embedding.from_pretrained(pre_train,
                                                          padding_idx=0)
        else:
            self.embedding = nn.Embedding(type_num, dim, padding_idx=0)

    def forward(self, g, p_name = 'node_type'):
        """Input type is dgl graph"""
        nnode_feats = self.embedding(g.ndata.pop(p_name))
        return nnode_feats
    
class FakeAtomEmbedding(nn.Module):
    """
    Convert the atom(node) list to atom embeddings.
    The atom with the same element share the same initial embeddding.
    """

    def __init__(self, dim=128, type_num=300, pre_train=None):
        """
        Randomly init the element embeddings.
        Args:
            dim: the dim of embeddings
            type_num: the largest atomic number of atoms in the dataset
            pre_train: the pre_trained embeddings
        """
        super().__init__()
        self._dim = dim
        self._type_num = type_num
        if pre_train is not None:
            self.embedding = nn.Embedding.from_pretrained(pre_train,
                                                          padding_idx=0)
        else:
            self.embedding = nn.Embedding(type_num, dim, padding_idx=0)

    def forward(self, g, p_name = 'node_type'):
        """Input type is dgl graph"""
        node_type = g.ndata.pop(p_name) + 100 * g.ndata.pop("ls")
        nnode_feats = self.embedding(node_type)
        return nnode_feats

class GPEmbedding(nn.Module):
    """
    Convert the atom(node) list to group and period embeddings.
    The atom with the same element share the same initial embeddding.
    """

    def __init__(self, dim=128, type_group=18, type_period = 7, dim_ratio_group = 2/3,):
        """
        Randomly init the element embeddings.
        Args:
            dim: the dim of embeddings
            type_num: the largest atomic number of atoms in the dataset
            pre_train: the pre_trained embeddings
        """
        super().__init__()
        self._dim = dim
        self._type_group = type_group
        self._type_period = type_period
        self._dim_group = int(dim * dim_ratio_group)
        self._dim_period = dim - self._dim_group
        self.gembedding = nn.Embedding(type_group, self._dim_group, padding_idx=0)
        self.pembedding = nn.Embedding(type_period, self._dim_period, padding_idx=0)

    def forward(self, g, p_name="node"):
        """Input type is dgl graph"""
        group_list = g.ndata.pop("group")
        period_list =g.ndata.pop("period")
        gembed = self.gembedding(group_list)
        pembed = self.pembedding(period_list)
        g.ndata[p_name] = torch.cat((gembed, pembed), dim=1)
        return g.ndata[p_name]

class GPLSEmbedding(nn.Module):
    """
    Convert the atom(node) list to group, period and label site embeddings.
    The atom with the same element share the same initial embeddding.
    """

    def __init__(self, dim=128, 
                       type_group=18, 
                       type_period = 7, 
                       type_ls = 3,
                       dim_ratio_group = 1/2, 
                       dim_ratio_period = 1/4):
        """
        Randomly init the element embeddings.
        Args:
            dim:       the dim of embeddings
            type_num:  othe largest atomic number of atoms in the dataset
            pre_train: the pre_trained embeddings
        """
        super().__init__()
        self._dim = dim
        self._type_group = type_group
        self._type_period = type_period
        self._type_ls = type_ls
        # Set different dimension for group, period and LS should have big difference
        # Thus I suggess keep the LS as 0, 1 information
        self._dim_group = int(dim *dim_ratio_group )
        self._dim_period =  int(dim *dim_ratio_period)
        self._dim_ls = dim - self._dim_group - self._dim_period 
        self.gembedding = nn.Embedding(type_group, self._dim_group, padding_idx=0)
        self.pembedding = nn.Embedding(type_period,self._dim_period, padding_idx=0)
        self.lsembedding= nn.Embedding(type_ls, self._dim_ls, padding_idx=0)

    def forward(self, g, p_name="node"):
        """Input type is dgl graph"""
        group_list = g.ndata.pop("group")
        period_list =g.ndata.pop("period")
        ls_list = g.ndata.pop("ls")
        gembed = self.gembedding(group_list)
        pembed = self.pembedding(period_list)
        lsembed= self.lsembedding(ls_list)
        g.ndata[p_name] = torch.cat((gembed, pembed, lsembed), dim=1)
        return g.ndata[p_name]