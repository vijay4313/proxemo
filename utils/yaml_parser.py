import yaml
import os

def yaml_parser(file_name, config_base_path = '../runner/config'):
    file_path = os.path.join(config_base_path,
                             file_name + '.yaml' )
    with open(file_path, 'r') as yaml_file:
        yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return yaml_dict
        
        
if __name__ == '__main__':
    yaml_dict = yaml_parser('stgcn')
    print(yaml_dict)