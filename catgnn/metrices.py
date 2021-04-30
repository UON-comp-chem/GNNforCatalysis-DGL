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
    


__all__ = ['Meter']

# pylint: disable=E1101
class Meter(object):
    """Track and summarize model performance on a dataset for (multi-label) prediction.
    When dealing with multitask learning, quite often we normalize the labels so they are
    roughly at a same scale. During the evaluation, we need to undo the normalization on
    the predicted labels. If mean and std are not None, we will undo the normalization.
    Currently we support evaluation with 4 metrics:
    * ``pearson r2``
    * ``mae``
    * ``rmse``
    * ``roc auc score``
    Parameters
    ----------
    mean : torch.float32 tensor of shape (T) or None.
        Mean of existing training labels across tasks if not ``None``. ``T`` for the
        number of tasks. Default to ``None`` and we assume no label normalization has been
        performed.
    std : torch.float32 tensor of shape (T)
        Std of existing training labels across tasks if not ``None``. Default to ``None``
        and we assume no label normalization has been performed.
    Examples
    --------
    Below gives a demo for a fake evaluation epoch.
    >>> import torch
    >>> from dgllife.utils import Meter
    >>> meter = Meter()
    >>> # Simulate 10 fake mini-batches
    >>> for batch_id in range(10):
    >>>     batch_label = torch.randn(3, 3)
    >>>     batch_pred = torch.randn(3, 3)
    >>>     meter.update(batch_pred, batch_label)
    >>> # Get MAE for all tasks
    >>> print(meter.compute_metric('mae'))
    [1.1325558423995972, 1.0543707609176636, 1.094650149345398]
    >>> # Get MAE averaged over all tasks
    >>> print(meter.compute_metric('mae', reduction='mean'))
    1.0938589175542195
    >>> # Get the sum of MAE over all tasks
    >>> print(meter.compute_metric('mae', reduction='sum'))
    3.2815767526626587
    """
    def __init__(self, mean=None, std=None):
        self.mask = []
        self.y_pred = []
        self.y_true = []

        if (mean is not None) and (std is not None):
            self.mean = mean.cpu()
            self.std = std.cpu()
        else:
            self.mean = None
            self.std = None

    def update(self, y_pred, y_true, mask=None):
        """Update for the result of an iteration
        Parameters
        ----------
        y_pred : float32 tensor
            Predicted labels with shape ``(B, T)``,
            ``B`` for number of graphs in the batch and ``T`` for the number of tasks
        y_true : float32 tensor
            Ground truth labels with shape ``(B, T)``
        mask : None or float32 tensor
            Binary mask indicating the existence of ground truth labels with
            shape ``(B, T)``. If None, we assume that all labels exist and create
            a one-tensor for placeholder.
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        if mask is None:
            self.mask.append(torch.ones(self.y_pred[-1].shape))
        else:
            self.mask.append(mask.detach().cpu())

    def _finalize(self):
        """Prepare for evaluation.
        If normalization was performed on the ground truth labels during training,
        we need to undo the normalization on the predicted labels.
        Returns
        -------
        mask : float32 tensor
            Binary mask indicating the existence of ground
            truth labels with shape (B, T), B for batch size
            and T for the number of tasks
        y_pred : float32 tensor
            Predicted labels with shape (B, T)
        y_true : float32 tensor
            Ground truth labels with shape (B, T)
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        if len(y_pred.size()) == 1:
            y_pred = y_pred.reshape(-1, 1)
            y_true = y_true.reshape(-1, 1)
            mask = mask.reshape(-1, 1)

        if (self.mean is not None) and (self.std is not None):
            # To compensate for the imbalance between labels during training,
            # we normalize the ground truth labels with training mean and std.
            # We need to undo that for evaluation.
            y_pred = y_pred * self.std + self.mean

        return mask, y_pred, y_true

    def _reduce_scores(self, scores, reduction='none'):
        """Finalize the scores to return.
        Parameters
        ----------
        scores : list of float
            Scores for all tasks.
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks
        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        if reduction == 'none':
            return scores
        elif reduction == 'mean':
            return np.mean(scores)
        elif reduction == 'sum':
            return np.sum(scores)
        else:
            raise ValueError(
                "Expect reduction to be 'none', 'mean' or 'sum', got {}".format(reduction))

    def multilabel_score(self, score_func, reduction='none'):
        """Evaluate for multi-label prediction.
        Parameters
        ----------
        score_func : callable
            A score function that takes task-specific ground truth and predicted labels as
            input and return a float as the score. The labels are in the form of 1D tensor.
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks
        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        mask, y_pred, y_true = self._finalize()
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            task_score = score_func(task_y_true, task_y_pred)
            if task_score is not None:
                scores.append(task_score)
        return self._reduce_scores(scores, reduction)

    def pearson_r2(self, reduction='none'):
        """Compute squared Pearson correlation coefficient.
        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks
        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        def score(y_true, y_pred):
            return pearsonr(y_true.numpy(), y_pred.numpy())[0] ** 2
        return self.multilabel_score(score, reduction)

    def mae(self, reduction='none'):
        """Compute mean absolute error.
        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks
        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        def score(y_true, y_pred):
            return F.l1_loss(y_true, y_pred).data.item()
        
        return self.multilabel_score(score, reduction)

    def rmse(self, reduction='none'):
        """Compute root mean square error.
        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks
        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        def score(y_true, y_pred):
            return torch.sqrt(F.mse_loss(y_pred, y_true).cpu()).item()
        return self.multilabel_score(score, reduction)

    def roc_auc_score(self, reduction='none'):
        """Compute the area under the receiver operating characteristic curve (roc-auc score)
        for binary classification.
        ROC-AUC scores are not well-defined in cases where labels for a task have one single
        class only (e.g. positive labels only or negative labels only). In this case we will
        simply ignore this task and print a warning message.
        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks.
        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        # Todo: This function only supports binary classification and we may need
        #  to support categorical classes.
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'
        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'ROC AUC score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                return roc_auc_score(y_true.long().numpy(), torch.sigmoid(y_pred).numpy())
        return self.multilabel_score(score, reduction)

    def pr_auc_score(self, reduction='none'):
        """Compute the area under the precision-recall curve (pr-auc score)
        for binary classification.
        PR-AUC scores are not well-defined in cases where labels for a task have one single
        class only (e.g. positive labels only or negative labels only). In this case, we will
        simply ignore this task and print a warning message.
        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks.
        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'
        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'PR AUC score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                precision, recall, _ = precision_recall_curve(
                    y_true.long().numpy(), torch.sigmoid(y_pred).numpy())
                return auc(recall, precision)
        return self.multilabel_score(score, reduction)

    def compute_metric(self, metric_name, reduction='none'):
        """Compute metric based on metric name.
        Parameters
        ----------
        metric_name : str
            * ``'r2'``: compute squared Pearson correlation coefficient
            * ``'mae'``: compute mean absolute error
            * ``'rmse'``: compute root mean square error
            * ``'roc_auc_score'``: compute roc-auc score
            * ``'pr_auc_score'``: compute pr-auc score
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks
        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        if metric_name == 'r2':
            return self.pearson_r2(reduction)
        elif metric_name == 'mae':
            return self.mae(reduction)
        elif metric_name == 'rmse':
            return self.rmse(reduction)
        elif metric_name == 'roc_auc_score':
            return self.roc_auc_score(reduction)
        elif metric_name == 'pr_auc_score':
            return self.pr_auc_score(reduction)
        else:
            raise ValueError('Expect metric_name to be "r2" or "mae" or "rmse" '
                             'or "roc_auc_score" or "pr_auc", got {}'.format(metric_name))


def mae_loss(y_pred, y_true):
    err = torch.abs(y_true - y_pred)
    mae = err.mean()
    return mae