#!/usr/bin/env python
# Title           :loader.py
# Author          :Venkatraman Narayanan, Bala Murali Manoghar, Vishnu Shashank Dorbala, Aniket Bera, Dinesh Manocha
# Copyright       :"Copyright 2020, Proxemo project"
# Version         :1.0
# License         :"MIT"
# Maintainer      :Venkatraman Narayanan, Bala Murali Manoghar
# Email           :vnarayan@terpmail.umd.edu, bsaisudh@terpmail.umd.edu
#==============================================================================
import math
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchlight


def weights_init(m):
    """Random initialize weights

    Args:
        m (torch): torch layer object
    """
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def find_all_substr(a_str, sub):
    """Find substring

    Args:
        a_str (str): Actual string
        sub (str): subtring to be matched
    """
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


def get_best_epoch_and_accuracy(path_to_model_files):
    """Get epoch with best accuracy.

    Args:
        path_to_model_files (str): Model summary/checkpoints path

    Returns:
        [list]: Best epoch and corresponding accuracy
    """
    all_models = os.listdir(path_to_model_files)
    while '_' not in all_models[-1]:
        all_models = all_models[:-1]
    best_model = all_models[-1]
    all_us = list(find_all_substr(best_model, '_'))
    return int(best_model[5:all_us[0]]), float(best_model[all_us[0]+4:all_us[1]])


def get_optimizer(optimizer_name):
    """Get Optimizer based on name.

    Args:
        optimizer_name (str): Optimizer name

    Raises:
        ValueError: If optimizer type is unavailable

    Returns:
        [Torch.optim]: Optimizer object
    """
    if optimizer_name == "sgd":
        return optim.SGD
    elif optimizer_name == "adam":
        return optim.Adam
    else:
        raise ValueError('Unknown Optimizer ' + optimizer_name)


def get_loss_fn(loss_name):
    """Retrieve loss function.

    Args:
        loss_name (str): loss function name

    Raises:
        ValueError: If loss type is unavailable

    Returns:
        [obj]: loss function
    """
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError('Unknown loss function')


class SummaryStatistics(object):
    """Generate train/test summary stats
    Tracks the following metrics:
    - Confusion Matrix
    - Average & Per-class precision
    - Average & Per-class recall
    - Average & Per-class acuuracy
    - Average & Per-class f1-score
    """

    def __init__(self, n_classes=4):
        """Constructor

        Args:
            n_classes (int, optional): Number of output classes. Defaults to 4.
        """
        self.n_classes = n_classes
        self.reset()

    def reset(self):
        """Reset the stats."""
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def update(self, true_labels, pred_labels):
        """Update the confusion matrix and metrics

        Args:
            true_labels (np.array): Actual labels
            pred_labels (np.array): Predicted labels
        """
        if len(pred_labels.shape) > 1:
            pred_labels = np.argmax(pred_labels, axis=-1)
        conf_matrix = np.bincount(
            self.n_classes * true_labels.astype(int) + pred_labels.astype(int),
            minlength=self.n_classes ** 2
        ).reshape(self.n_classes, self.n_classes)

        self.confusion_matrix += conf_matrix

    def get_metrics(self):
        """Generate/Retrieve the summary metrics.

        Returns:
            [dict]: All metrics mentioned above.
        """
        conf_matrix = self.confusion_matrix
        precision_per_class = np.nan_to_num(
            np.diag(conf_matrix) / np.sum(conf_matrix, axis=0))
        recall_per_class = np.nan_to_num(
            np.diag(conf_matrix) / np.sum(conf_matrix, axis=1))
        acc_per_class = np.nan_to_num(np.diag(conf_matrix) / (np.sum(
            conf_matrix, axis=1) + np.sum(conf_matrix, axis=0) - np.diag(conf_matrix)))
        f1_per_class = np.nan_to_num(
            2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class))

        avg_precision = np.nanmean(precision_per_class)
        avg_recall = np.nanmean(recall_per_class)
        avg_acc = np.nanmean(acc_per_class)
        avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)

        result = {
            'conf_matrix':  conf_matrix,
            'stats_per_class':  {
                'class_precision':  precision_per_class,
                'class_recall': recall_per_class,
                'class_accuracy':   acc_per_class,
                'class_f1': f1_per_class
            },
            'avg_stats': {
                'avg_precision':    avg_precision,
                'avg_recall':   avg_recall,
                'avg_accuracy': avg_acc,
                'avg_f1':   avg_f1
            }
        }

        return result
