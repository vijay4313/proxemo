#!/usr/bin/env python
# Title           :loader.py
# Author          :Venkatraman Narayanan, Bala Murali Manoghar, Vishnu Shashank Dorbala, Aniket Bera, Dinesh Manocha
# Copyright       :"Copyright 2020, Proxemo project"
# Version         :1.0
# License         :"MIT"
# Maintainer      :Venkatraman Narayanan, Bala Murali Manoghar
# Email           :vnarayan@terpmail.umd.edu, bsaisudh@terpmail.umd.edu
#==============================================================================
import yaml
import os
import shutil
import sys


def yaml_parser(file_name, config_base_path='../modeling/config'):
    """YAML file parser.

    Args:
        file_name (str): YAML file to be loaded
        config_base_path (str, optional): Directory path of file.
                                          Defaults to '../modeling/config'.

    Returns:
        [dict]: Parsed YAML file as dictionary
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    config_base_path = os.path.normpath(os.path.join(cur_dir, config_base_path))

    file_path = os.path.join(config_base_path,
                             file_name + '.yaml')
    with open(file_path, 'r') as yaml_file:
        yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return yaml_dict


def copy_yaml(file_name, dest_folder, config_base_path='../modeling/config'):
    """Copies YAML file from one folder to another.

    Args:
        file_name (str): source file path
        dest_folder (str): destination path
        config_base_path (str, optional): Source file path. Defaults to '../modeling/config'.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    config_base_path = os.path.normpath(os.path.join(cur_dir, config_base_path))

    file_path = os.path.join(config_base_path,
                             file_name + '.yaml')
    try:
        shutil.copy(file_path, dest_folder)
    except IOError as e:
        print("Unable to copy yaml file. %s" % e)
    except:
        print("Unexpected error while copying yaml file:", sys.exc_info())


if __name__ == '__main__':
    yaml_dict = yaml_parser('train')
    print(yaml_dict)
