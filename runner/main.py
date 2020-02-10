import argparse
import os

import numpy as np
import torch
import torchlight
import yaml

from loader import loader
from trainer import Trainer

# base_path = os.path.dirname(os.path.realpath(__file__))
# data_path = os.path.join(base_path, '../data')
# ftype = 'Combined2'
# coords = 3
# joints = 16
# cycles = 1
# model_path = os.path.join(base_path, 'model_classifier_stgcn/features'+ftype)

# Load settings
parser = argparse.ArgumentParser(description='Gait Gen')
parser.add_argument('--settings', type=str, metavar='s',
                    help='config file for running the network.')
args = parser.parse_args()
model_args = args['MODEL']
data_args = args["DATA"]

# Load datasets
data, labels, data_train, labels_train, data_test, labels_test =\
    loader.load_data(data_args['DATA_PATH'], ftype,
                     data_args['COORDS'],
                     data_args['JOINTS'],
                     cycles=data_args['CYCLES'])
num_classes = np.unique(labels_train).shape[0]

# Convert datasets to Pytorch data

data_loader_train_test = {
    "train": torch.utils.data.DataLoader(
        dataset=loader.TrainTestLoader(
            data_train, labels_train, data_args['JOINTS'], data_args['COORDS'], num_classes),
        batch_size=args['BATCH_SIZE'],
        shuffle=True,
        num_workers=args['NUM_WORKERS'],
        drop_last=True),
    "test": torch.utils.data.DataLoader(
        dataset=loader.TrainTestLoader(
            data_test, labels_test, data_args['JOINTS'], data_args['COORDS'], num_classes),
        batch_size=args['BATCH_SIZE'],
        shuffle=True,
        num_workers=args['NUM_WORKERS'],
        drop_last=True)}

graph_dict = {'strategy': 'spatial'}

# Build model
model = Trainer(args, data_loader_train_test,
                num_classes, graph_dict)

# Run train/test loop
if args.train:
    model.train()
if args.save_features:
    f = model.save_best_feature(ftype, data, joints, coords)
