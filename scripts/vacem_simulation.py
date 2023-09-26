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


def run_grid_scan(yaml_file, save_path, geometry='xz',
                  low_memory_mode=False, n_threads=12):
    params_to_vary, data = decypher_yaml(yaml_file)
    laser_params, factors, resolutions = data
    
    # Iterate over variable parameters and create appropriate laser_params
    # Run simulations: should we include postprocess to run_simulation()?
    
    
    
    
    

