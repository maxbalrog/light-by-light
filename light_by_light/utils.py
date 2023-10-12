'''
General utility functions

Author: Maksim Valialshchikov, @maxbalrog (github)
'''
import yaml
import numpy as np

__all__ = ['read_yaml', 'write_yaml', 'decypher_yaml', 'write_generic_yaml',
           ]

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


def get_grid_from_list(data):
    '''
    It is expected that parameter lists would be of the following type:
        - [start, end, npts]
        - [start, end, npts, scale]
    '''
    start, end, npts = data[:3]
    step = (end - start) / (npts - 1)
    grid = np.linspace(start, end, npts, endpoint=True)
    if np.isclose(step, int(np.round(step))):
        grid = grid.astype(int)
    result = grid if len(data) == 3 else (grid, data[3])
    return result


def collect_study_data(study):
    '''
    Collect study data (optimization params and user attrs) to one dictionary
    '''
    trials = [trial for trial in study.trials if trial.values is not None]
    n = len(trials)
    assert n > 0
    param_keys = list(trials[0].params.keys())
    user_attr_keys = list(trials[0].user_attrs.keys())
    keys = param_keys + user_attr_keys
    data = {key: np.empty(n) for key in keys}
    # params = {param: np.empty(n) for param in trials[0].params}
    # user_attrs = {param: np.empty(n) for param in trials[0].user_attrs}
    for i,trial in enumerate(trials):
        for key in param_keys:
            data[key][i] = trial.params[key]
        for key in user_attr_keys:
            data[key][i] = trial.user_attrs[key]
    return data
        
    
