'''
General utility functions

Author: Maksim Valialshchikov, @maxbalrog (github)
'''
import yaml
import numpy as np

__all__ = ['read_yaml', 'write_yaml', 'decypher_yaml', 'write_generic_yaml',
           'get_grids_from_dict', 'decypher_yaml_grid']

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
        

def write_generic_yaml(yaml_file, laser_params, simbox_params, mode='gridscan'):
    data = {}
    data['_mode'] = mode
    data['lasers'] = {f'laser_{i}': param for i,param in enumerate(laser_params)}
    data['simbox_params'] = simbox_params 
    write_yaml(yaml_file, data)
        

def get_grids_from_dict(data_dict):
    '''
    It is expected that parameter lists would be of the following type:
        - [start, end, npts]
        - [start, end, npts, scale]
    '''
    vary = {}
    for key,value in data_dict.items():
        if type(value) in [list, tuple]:
            grid = np.linspace(value[0], value[1], value[2], endpoint=True)
            if len(value) == 3:
                vary[key] = grid
            elif len(value) == 4:
                # value[3] is scale
                vary[key] = (grid, value[3])
    return vary


def decypher_yaml_grid(yaml_file):
    data = read_yaml(yaml_file)
    simbox_params = data['simbox_params']
    lasers = data['lasers']
    laser_params = [params for params in lasers.values()]
    simbox_vary, lasers_vary = {}, {}
    
    for key in simbox_params.keys():
        simbox_vary[key] = get_grids_from_dict(simbox_params[key])
    for key in lasers.keys():
        lasers_vary[key] = get_grids_from_dict(lasers[key])
    return (lasers_vary, simbox_vary), (laser_params, simbox_params)
        
    
