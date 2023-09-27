'''
Utility functions for running vacem simulations

Author: Maksim Valialshchikov, @maxbalrog (github)
'''
import os
from pathlib import Path

from scripts.vacem_ini import create_ini_file
from scripts.utils import decypher_yaml

__all__ = ['run_simulation']

vacem_path = '/home/wi73yus/packages/vacem-master/scripts/vacem_solver.py'


def run_simulation(laser_params, save_path, factors, resolutions,
                   geometry='xz', low_memory_mode=False, n_threads=12):
    # Make sure the directory exists
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    
    # Create .ini file
    create_ini_file(laser_params, save_path, factors, resolutions,
                    geometry, low_memory_mode)
    vacem_ini = f'{save_path}/vacem.ini'
    
    # Run vacem script
    os.system(f'{vacem_path} --output {save_path} --threads {n_threads} load_ini {vacem_ini}')
    return 0


def check_grids(params_to_vary):
    lasers_vary = params_to_vary['lasers']
    n_factors_vary = len(params_to_vary['factors'].keys())
    n_resolutions_vary = len(params_to_vary['resolutions'].keys())
    n_lasers_vary = sum([len(lasers_vary[laser].keys()) for laser in lasers_vary.keys()])
    assert n_factors_vary < 2, "Only 1d grid scan is supported for factors"
    assert n_resolutions_vary < 2, "Only 1d grid scan is supported for resolutions"
    assert n_lasers_vary < 2, "Only 1d grid scan is supported for lasers"
    assert n_factors_vary+n_resolutions_vary+n_lasers_vary < 2, "Only 1d grid scan is supported"
    
    return n_factors_vary, n_resolutions_vary, n_lasers_vary


def run_grid_scan(yaml_file, save_path, geometry='xz',
                  low_memory_mode=False, n_threads=12):
    params_to_vary, data = decypher_yaml(yaml_file)
    laser_params, factors, resolutions = data
    lasers_vary = params_to_vary['lasers']
    
    n_factors_vary, n_resolutions_vary, n_lasers_vary = check_grids(params_to_vary)
    
    if n_factors_vary > 0:
        key, grid = list(params_to_vary['factors'].items())[0]
        save_path = f'{os.path.dirname(save_path)}/{key}/'
        for value in grid:
            factors[key] = 
        
    
    # Iterate over variable parameters and create appropriate laser_params
    # Run simulations: should we include postprocess to run_simulation()?
    
    
    
    
    

