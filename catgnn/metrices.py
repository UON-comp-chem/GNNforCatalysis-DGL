#!/usr/bin/env python
# coding: utf-8
# Author SchNetPack developers
# License The MIT License

"""
Shameless hacked from https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/train/metrics.py
Credit goes to SchNetPack developers
"""

import warnings
import numpy as np
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import warnings


class Metric:
    r"""
    Base class for all metrics.

    Metrics measure the performance during the training and evaluation.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
            `MSE_[target]` will be used (Default: None)
        element_wise (bool): set to True if the model output is an element-wise
            property (forces, positions, ...)
    """

    def __init__(self, target, model_output=None, name=None, element_wise=False):
        self.target = target
        self.model_output = target if model_output is None else model_output
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.element_wise = element_wise

    def add_batch(self, batch, result):
        """ Add a batch to calculate the metric on """
        raise NotImplementedError

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        raise NotImplementedError

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""
        pass
    
class MeanAbsoluteError(Metric):
    r"""
    Metric for mean absolute error. For non-scalar quantities, the mean of all
    components is taken.
    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
            `MAE_[target]` will be used (Default: None)
        element_wise (bool): set to True if the model output is an element-wise
            property (forces, positions, ...)
    """

    def __init__(
        self,
        target,
        model_output=None,
        bias_correction=None,
        name=None,
        element_wise=False,
    ):
        name = "MAE_" + target if name is None else name
        super(MeanAbsoluteError, self).__init__(
            target=target,
            model_output=model_output,
            name=name,
        )

        self.bias_correction = bias_correction

        self.l1loss = 0.0
        self.n_entries = 0.0

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""
        self.l1loss = 0.0
        self.n_entries = 0.0

    def _get_diff(self, y, yp):
        diff = y - yp
        if self.bias_correction is not None:
            diff += self.bias_correction
        return diff

    def add_batch(self, batch, result):
        y = batch.label
        if self.model_output is None:
            yp = result
        else:
            if type(self.model_output) is list:
                for idx in self.model_output:
                    result = result[idx]
                    # print(result.shape)
            else:
                result = result
            yp = result

        # print(yp, yp.shape, y.shape)
        diff = self._get_diff(y, yp)
        # print(diff)
        # print()
        self.l1loss += (
            torch.sum(torch.abs(diff).view(-1), 0).detach().cpu().data.numpy()
        )
        
        self.n_entries += np.prod(y.shape)

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        return self.l1loss / self.n_entries
    
