import math
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchlight


def weights_init(m):
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
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


def get_best_epoch_and_accuracy(path_to_model_files):
    all_models = os.listdir(path_to_model_files)
    while '_' not in all_models[-1]:
        all_models = all_models[:-1]
    best_model = all_models[-1]
    all_us = list(find_all_substr(best_model, '_'))
    return int(best_model[5:all_us[0]]), float(best_model[all_us[0]+4:all_us[1]])


def get_optimizer(optimizer_name):
    if optimizer_name == "sgd":
        return optim.SGD
    elif optimizer_name == "adam":
        return optim.Adam
    else:
        raise ValueError('Unknown Optimizer ' + optimizer_name)


def get_loss_fn(loss_name):
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError('Unknown loss function')

