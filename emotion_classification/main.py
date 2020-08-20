#!/usr/bin/env python
# Title           :loader.py
# Author          :Venkatraman Narayanan, Bala Murali Manoghar, Vishnu Shashank Dorbala, Aniket Bera, Dinesh Manocha
# Copyright       :"Copyright 2020, Proxemo project"
# Version         :1.0
# License         :"MIT"
# Maintainer      :Venkatraman Narayanan, Bala Murali Manoghar
# Email           :vnarayan@terpmail.umd.edu, bsaisudh@terpmail.umd.edu
#==============================================================================
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
    """CLI arg parser.

    Returns:
        [dict]: CLI args
    """
    parser = argparse.ArgumentParser(description='Proxemo Runner')
    parser.add_argument('--settings', type=str, default='infer', metavar='s',
                        help='config file for running the network.')
    cli_args = parser.parse_args()

    args = yaml_parser.yaml_parser(cli_args.settings)

    return args


def main():
    """Main routine."""
    args = arg_parser()
    gen_args, model_args, data_args = args.values()
    # Build model
    model = Trainer(gen_args, data_args, model_args)

    if gen_args['MODE'] == 'train':
        model.train()

    elif gen_args['MODE'] == 'test':
        model.test()


if __name__ == '__main__':
    main()
