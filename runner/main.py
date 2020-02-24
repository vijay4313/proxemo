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


# python main.py --settings stgcn

# Load settings
parser = argparse.ArgumentParser(description='Gait Gen')
parser.add_argument('--settings', type=str, default='vs_gcnn', metavar='s',
                    help='config file for running the network.')
cli_args = parser.parse_args()
print(f'---> Settings file - {cli_args.settings}')
args = yaml_parser.yaml_parser(cli_args.settings)
model_args = args['MODEL']
data_args = args["DATA"]
args['YAML_FILE_NAME'] = cli_args.settings

if args['MODE'] == 'train':
    test_size = 0.1
else:
    test_size = 1.0

# Load datasets
if data_args['TYPE'] == 'single_view':
    data, labels, data_train, labels_train, data_test, labels_test =\
        loader.load_data(data_args['FEATURES_FILE'],
                         data_args['LABELS_FILE'],
                         data_args['COORDS'],
                         data_args['JOINTS'],
                         cycles=data_args['CYCLES'],
                         test_size = test_size)
    num_classes_label = np.unique(labels_train).shape[0]

# Load datasets multiview
if data_args['TYPE'] == 'multi_view':
    # load Data
    data, labels, \
    data_train, labels_train, \
    data_test, labels_test , \
    angles_train, angles_test = \
        loader.load_data_multiview(data_args['FEATURES_FILE'],
                                   data_args['LABELS_FILE'],
                                   data_args['COORDS'],
                                   data_args['JOINTS'],
                                   cycles=data_args['CYCLES'],
                                   test_size = test_size)
        
    # convert to view group (4 view groups)
    angles_train = list((np.asarray(angles_train)/90).astype(int))
    angles_test = list((np.asarray(angles_test)/90).astype(int))
    
    # number of classes
    num_classes_label = np.unique(labels_train).shape[0]
    num_classes_angles = np.unique(angles_train).shape[0]

# Convert datasets to Pytorch data and model specific parameters

if model_args['TYPE'] in ['stgcn', 'vscnn']:
    num_classes = num_classes_label
    model_kwargs = {}
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
            
if model_args['TYPE'] == 'vscnn_vgp':
    num_classes = num_classes_angles
    model_kwargs = {}
    print(f"---> num classes : {num_classes}")
    data_loader_train_test = {
        "train": torch.utils.data.DataLoader(
            dataset=loader.TrainTestLoader_vscnn(
                data_train, angles_train, 
                data_args['JOINTS'], data_args['COORDS'],
                num_classes),
            batch_size=args['BATCH_SIZE'],
            shuffle=True,
            num_workers=args['NUM_WORKERS'],
            drop_last=True),
        "test": torch.utils.data.DataLoader(
            dataset=loader.TrainTestLoader_vscnn(
                data_test, angles_test,
                data_args['JOINTS'], data_args['COORDS'],
                num_classes),
            batch_size=args['BATCH_SIZE'],
            shuffle=True,
            num_workers=args['NUM_WORKERS'],
            drop_last=True)}
            
if model_args['TYPE'] in ['vscnn_vgf', 'vs_gcnn']:
    num_classes = num_classes_label
    model_kwargs = {'NUM_GROUPS' : num_classes_angles}
    print(f"---> num classes : {num_classes}")
    data_loader_train_test = {
        "train": torch.utils.data.DataLoader(
            dataset=loader.TrainTestLoader_vscnn(
                data_train, list(zip(labels_train, angles_train)), 
                data_args['JOINTS'], data_args['COORDS'],
                num_classes),
            batch_size=args['BATCH_SIZE'],
            shuffle=True,
            num_workers=args['NUM_WORKERS'],
            drop_last=True),
        "test": torch.utils.data.DataLoader(
            dataset=loader.TrainTestLoader_vscnn(
                data_test, list(zip(labels_test, angles_test)),
                data_args['JOINTS'], data_args['COORDS'],
                num_classes),
            batch_size=args['BATCH_SIZE'],
            shuffle=True,
            num_workers=args['NUM_WORKERS'],
            drop_last=True)}

graph_dict = {'strategy': 'spatial'}

# Build model
if model_args['TYPE'] == 'vscnn':
    model_vgp = Trainer(args, data_loader_train_test,
                        num_classes, graph_dict, model_kwargs)
    model_vgf = Trainer(args, data_loader_train_test,
                        num_classes, graph_dict, model_kwargs)
else:
    model = Trainer(args, data_loader_train_test,
                    num_classes, graph_dict, model_kwargs)

# Run train/test loop
if args['MODE'] == 'train':
    if args['WARM_START']:
        print('---> Warm Start')
        model.warm_start()
    model.train()
if args['MODE'] == 'test':
    if model_args['TYPE'] == 'vscnn':
        pass
    else:
        model.load_model()
        model.test()
#if args['SAVE_FEATURES']:
#    f = model.save_best_feature(data_args['FEATURES_FILE'],
#                                args['SAVE_FILE'],
#                                data,
#                                data_args['JOINTS'],
#                                data_args['COORDS'])
