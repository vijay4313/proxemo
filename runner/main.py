import sys
sys.path.append("../")
import argparse
import os

import numpy as np
import torch
import torchlight


from loader import loader
from trainer import Trainer
from utils import yaml_parser

# base_path = os.path.dirname(os.path.realpath(__file__))
# data_path = os.path.join(base_path, '../data')
# ftype = 'Combined2'
# coords = 3
# joints = 16
# cycles = 1
# model_path = os.path.join(base_path, 'model_classifier_stgcn/features'+ftype)

# python main.py --settings stgcn

# Load settings
parser = argparse.ArgumentParser(description='Gait Gen')
parser.add_argument('--settings', type=str, default='stgcn', metavar='s',
                    help='config file for running the network.')
cli_args = parser.parse_args()
args = yaml_parser.yaml_parser(cli_args.settings)
model_args = args['MODEL']
data_args = args["DATA"]

# Load datasets
data, labels, data_train, labels_train, data_test, labels_test =\
    loader.load_data(data_args['FEATURES_FILE'],
                     data_args['LABELS_FILE'],
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
if args['MODE'] == 'train':
    model.train()
if args['SAVE_FEATURES']:
    f = model.save_best_feature(data_args['FEATURES_FILE'],
                                args['SAVE_FILE'],
                                data,
                                data_args['JOINTS'],
                                data_args['COORDS'])
