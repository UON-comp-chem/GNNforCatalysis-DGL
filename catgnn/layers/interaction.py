import torch
import torch.nn.functional as F
import dgl
import torch.nn as nn
from dgl.nn.pytorch.conv.cfconv import CFConv, ShiftedSoftplus

class SchInteraction(nn.Module):
    """Building block for SchNet.
    SchNet is introduced in `SchNet: A continuous-filter convolutional neural network for
    modeling quantum interactions <https://arxiv.org/abs/1706.08566>`__.
    This layer combines node and edge features in message passing and updates node
    representations.
    Parameters
    ----------
    node_feats : int
        Size for the input and output node features.
    edge_in_feats : int
        Size for the input edge features.
    """
    def __init__(self, edge_in_feats, node_feats):
        super(SchInteraction, self).__init__()

        self.conv = CFConv(node_feats, edge_in_feats, node_feats, node_feats)
        self.project_out = nn.Linear(node_feats, node_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for layer in self.conv.project_edge:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.conv.project_node.reset_parameters()
        self.conv.project_out[0].reset_parameters()
        self.project_out.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Performs message passing and updates node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feats)
            Input node features, V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features, E for the number of edges.
        Returns
        -------
        float32 tensor of shape (V, node_feats)
            Updated node representations.
        """
        node_feats = self.conv(g, node_feats, edge_feats)
        return self.project_out(node_feats)