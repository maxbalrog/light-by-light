'''
General utility functions

Author: Maksim Valialshchikov, @maxbalrog (github)
'''
import yaml
import numpy as np

__all__ = ['read_yaml', 'write_yaml', 'decypher_yaml']

def read_yaml(yaml_file):
    with open(yaml_file, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            return data
        except yaml.YAMLError as exc:
            print(exc)
            return exc


def write_yaml(yaml_file, data):
    with open(yaml_file, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
        

def get_grids_from_dict(data_dict):
    vary = {}
    for key,value in data_dict.items():
        if type(value) in [list, tuple]:
            # n_steps = int((value[1]-value[0])/value[2]) + 1
            vary[key] = np.linspace(value[0], value[1], value[2], endpoint=True)
    return vary


def decypher_yaml(yaml_file):
    data = read_yaml(yaml_file)
    factors = data['factors']
    resolutions = data['resolutions']
    lasers = data['lasers']
    laser_params = [params for params in lasers.values()]
    params_to_vary = {'factors': {},
                      'resolutions': {},
                      'lasers': {}}
    
    params_to_vary['factors'] = get_grids_from_dict(factors)
    params_to_vary['resolutions'] = get_grids_from_dict(resolutions)
    for laser_key in lasers.keys():
        params_to_vary['lasers'][laser_key] = get_grids_from_dict(lasers[laser_key])
    return params_to_vary, (laser_params, factors, resolutions)
        
    
