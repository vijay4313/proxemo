from utils import yaml_parser
from runner.trainer import Trainer
import torch
import numpy as np
import os
import argparse
import sys
sys.path.append("../")


# python main.py --settings stgcn

# Load settings

def arg_parser():
    parser = argparse.ArgumentParser(description='Proxemo Runner')
    parser.add_argument('--settings', type=str, default='stgcn', metavar='s',
                        help='config file for running the network.')
    cli_args = parser.parse_args()

    args = yaml_parser.yaml_parser(cli_args.settings)

    return args

def main():
    args = arg_parser()
    gen_args, model_args, data_args = args.values()
    # Build model
    model = Trainer(gen_args, data_args, model_args)

    if gen_args['MODE'] == 'train':
        model.train()

    elif args['MODE'] == 'test':
        model.test()

if __name__ = '__main__':
    main()