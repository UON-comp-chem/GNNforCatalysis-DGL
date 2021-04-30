import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl.function as fn
from torch.nn import Softplus
from dgl.nn.pytorch.conv.cfconv import CFConv, ShiftedSoftplus
from dgl.utils import expand_as_pair
import numpy as np

def softplus_inverse(x):
    return x + np.log(-np.expm1(-x))

class RBFLayer (nn.Module):
    r"""RBF Layer"""
    def __init__(self, low=0, high=10.0, num_gaussians=128):
        super(RBFLayer, self).__init__()
        offset = torch.linspace(low, high, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)
        self._fan_out = num_gaussians

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
    
    def reset_parameters(self):
        pass

class PhysRBFLayer(nn.Module):
    r"""RBF layer used in PhysNet"""
    def __init__(self, low = 0., high = 10., num_gaussians=64):
        super(PhysRBFLayer, self).__init__()
        self.num_gaussians = num_gaussians
        center = softplus_inverse(np.linspace(1.0,np.exp(-high),num_gaussians))
        width =  [softplus_inverse((0.5/((1.0-np.exp(-high))/num_gaussians))**2)]*num_gaussians
        self.register_buffer("high", torch.tensor(high, dtype=torch.float32))
        self.register_buffer(
            "center",
            torch.tensor(center)
            )
        self.register_buffer(
            "width",
            torch.tensor(width, dtype=torch.float32),
        )
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, r):
        rbf = torch.exp(-self.width * (torch.exp(-r.unsqueeze(-1)) - F.softplus(self.center)) ** 2)
        return rbf