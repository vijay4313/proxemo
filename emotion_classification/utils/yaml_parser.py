import yaml
import os
import shutil
import sys


def yaml_parser(file_name, config_base_path='../runner/config'):
    file_path = os.path.join(config_base_path,
                             file_name + '.yaml')
    with open(file_path, 'r') as yaml_file:
        yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return yaml_dict


def copy_yaml(file_name, dest_folder, config_base_path='../runner/config'):
    file_path = os.path.join(config_base_path,
                             file_name + '.yaml')
    try:
        shutil.copy(file_path, dest_folder)
    except IOError as e:
        print("Unable to copy yaml file. %s" % e)
    except:
        print("Unexpected error while copying yaml file:", sys.exc_info())


if __name__ == '__main__':
    yaml_dict = yaml_parser('stgcn')
    print(yaml_dict)
