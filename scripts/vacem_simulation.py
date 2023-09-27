'''
Utility functions for running vacem simulations

Author: Maksim Valialshchikov, @maxbalrog (github)
'''
import os
from pathlib import Path
import numpy as np

from vacem.support.eval_functions import polarization_vector

from scripts.vacem_ini import W_to_E0, create_ini_file
from scripts.utils import decypher_yaml_grid
from scripts.postprocess import SignalAnalyzer

__all__ = ['run_simulation', 'check_grids', 'run_gridscan_simbox',
           'run_gridscan_laser', 'run_gridscan']

vacem_path = '/home/wi73yus/packages/vacem-master/scripts/vacem_solver.py'


def run_simulation(laser_params, save_path, simbox_params,
                   geometry='xz', low_memory_mode=False, n_threads=12):
    # Make sure the directory exists
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    
    # Create .ini file
    create_ini_file(laser_params, save_path, simbox_params,
                    geometry, low_memory_mode)
    vacem_ini = f'{save_path}/vacem.ini'
    
    # Run vacem script
    os.system(f'{vacem_path} --output {save_path} --threads {n_threads} load_ini {vacem_ini}')
    return 0


def check_grids(lasers_vary, simbox_vary):
    n_lasers_vary = [len(lasers_vary[key].keys()) for key in lasers_vary.keys()]
    n_simbox_vary = [len(simbox_vary[key].keys()) for key in simbox_vary.keys()]
    assert sum(n_lasers_vary)+sum(n_simbox_vary) < 2, "Only 1d grid scans are supported"
    
    return n_lasers_vary, n_simbox_vary
        

def run_gridscan_simbox(simbox_vary, simbox_params, n_simbox_vary, laser_params,
                        save_path, geometry='xz', low_memory_mode=False,
                        n_threads=12, pol_idx=0, eps=1e-10):
    idx = np.argmax(np.array(n_simbox_vary))
    simbox_key = list(simbox_vary.keys())[idx]

    param_key, param_grid = list(simbox_vary[simbox_key].items())[0]
    save_path = f'{os.path.dirname(save_path)}/{simbox_key}_{param_key}/'
    for param_value in param_grid:
        simbox_params[simbox_key][param_key] = param_value
        if type(param_value) is int:
            save_folder = '{}{}_{}/'.format(save_path, param_key, param_value)
        else:
            save_folder = '{}{}_{:.1f}/'.format(save_path, param_key, param_value)

        run_simulation(laser_params, save_folder, simbox_params,
                       geometry, low_memory_mode, n_threads)

        theta = laser_params[pol_idx]['theta'] + eps
        phi = laser_params[pol_idx]['phi'] + eps
        beta = laser_params[pol_idx]['beta']
        laser_pol = polarization_vector(theta/180*np.pi,
                                        phi/180*np.pi,
                                        beta/180*np.pi)
        vacem_file = f'{os.path.dirname(save_folder)}/_vacem.npz'
        signal_analyzer = SignalAnalyzer(vacem_file, laser_pol, laser_params, geometry)
        signal_analyzer.get_discernible_signal()
        signal_analyzer.save_data(save_folder)


def run_gridscan_laser(lasers_vary, laser_params, n_lasers_vary, simbox_params,
                       save_path, geometry='xz', low_memory_mode=False,
                       n_threads=12, pol_idx=0, eps=1e-10):
    idx = np.argmax(np.array(n_lasers_vary))
    laser_key = list(lasers_vary.keys())[idx]

    scale = 1
    param_key, param_grid = list(lasers_vary[laser_key].items())[0]
    if type(param_grid) in [list, tuple]:
        scale = param_grid[1]
        param_grid = param_grid[0]
        
    save_path = f'{os.path.dirname(save_path)}/laser_{idx}_{param_key}/'
    for param_value in param_grid:
        save_folder = '{}{}_{:.1f}/'.format(save_path, param_key, param_value)
        laser_params[idx][param_key] = param_value * scale
        laser_params[idx]['E0'] = W_to_E0(laser_params[idx])

        run_simulation(laser_params, save_folder, simbox_params,
                       geometry, low_memory_mode, n_threads)

        theta = laser_params[pol_idx]['theta'] + eps
        phi = laser_params[pol_idx]['phi'] + eps
        beta = laser_params[pol_idx]['beta']
        laser_pol = polarization_vector(theta/180*np.pi,
                                        phi/180*np.pi,
                                        beta/180*np.pi)
        vacem_file = f'{os.path.dirname(save_folder)}/_vacem.npz'
        signal_analyzer = SignalAnalyzer(vacem_file, laser_pol, laser_params, geometry)
        signal_analyzer.get_discernible_signal()
        signal_analyzer.save_data(save_folder)


def run_gridscan(yaml_file, save_path, geometry='xz',
                 low_memory_mode=False, n_threads=12):
    (lasers_vary, simbox_vary), (laser_params, simbox_params) = decypher_yaml_grid(yaml_file)
    n_lasers_vary, n_simbox_vary = check_grids(lasers_vary, simbox_vary)
    
    if sum(n_simbox_vary) > 0:
        run_gridscan_simbox(simbox_vary, simbox_params, n_simbox_vary, laser_params,
                            save_path, geometry, low_memory_mode, n_threads, pol_idx=0)
    elif sum(n_lasers_vary) > 0:
        run_gridscan_laser(lasers_vary, laser_params, n_lasers_vary, simbox_params,
                           save_path, geometry, low_memory_mode, n_threads, pol_idx=0)
    print('Grid simulation finished!')
    
    
    
    
    

