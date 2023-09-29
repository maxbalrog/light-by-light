'''
Utility functions for running vacem simulations

Author: Maksim Valialshchikov, @maxbalrog (github)
'''
import os
from pathlib import Path
import numpy as np

from vacem.support.eval_functions import polarization_vector

from light_by_light.vacem_ini import W_to_E0, create_ini_file
from light_by_light.utils import read_yaml, get_grid_from_list
from light_by_light.postprocess import SignalAnalyzer

__all__ = ['run_simulation', 'run_simulation_postprocess', 'run_gridscan']

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
    return 1


def run_simulation_postprocess(laser_params, save_path, simbox_params,
                               geometry='xz', low_memory_mode=False, n_threads=12,
                               pol_idx=0, eps=1e-10):
    # Make sure the directory exists
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    
    # Create .ini file
    create_ini_file(laser_params, save_path, simbox_params,
                    geometry, low_memory_mode)
    vacem_ini = f'{save_path}/vacem.ini'
    
    # Run vacem script
    os.system(f'{vacem_path} --output {save_path} --threads {n_threads} load_ini {vacem_ini}')
    
    # Simulation result postprocessing (eps for stability)
    theta = laser_params[pol_idx]['theta'] + eps
    phi = laser_params[pol_idx]['phi'] + eps
    beta = laser_params[pol_idx]['beta']
    laser_pol = polarization_vector(theta/180*np.pi,
                                    phi/180*np.pi,
                                    beta/180*np.pi)
    vacem_file = f'{os.path.dirname(save_path)}/_vacem.npz'
    signal_analyzer = SignalAnalyzer(vacem_file, laser_pol, laser_params, geometry)
    signal_analyzer.get_discernible_signal()
    signal_analyzer.save_data(save_path)
    return 1


def run_gridscan(default_yaml, vary_yaml, save_path, eps=1e-10):
    '''
    Using vary_yaml file, determine variable parameter and perform 1D grid scan over it.
    Variable parameter could be found either in laser parameters or simbox parameters 
    (for benchmarking purposes).
    
    default_yaml: [str] - path to yaml file with simulation parameters
    vary_yaml: [str] - path to yaml file with parameters to vary
    save_path: [str] - directory where simulation results would be saved
    geometry, low_memory_mode, n_threads - params for run_simulation()
    pol_idx: laser polarization index [int] - which laser's polarization vector to take
                                              for calculation of Nperp
    eps: [float] - parameter for stability (for theta=0,180) when transfering to spherical
                   coordinate system
    '''
    default_params = read_yaml(default_yaml)
    params_vary = read_yaml(vary_yaml)
    
    # Define simulation parameters
    geometry = default_params['geometry']
    low_memory_mode = default_params['low_memory_mode']
    n_threads = default_params['n_threads']
    pol_idx = default_params['pol_idx']
    
    # Determine which parameter to vary
    key = list(params_vary.keys())[0]
    section_key = list(params_vary[key].keys())[0]

    # Extract grid and scale
    # With laser pulse parameters we can introduce scale since spatio-temporal scales
    # are small, e.g. tau ~ 20 fs. The grid is formed with dimensionless parameters and
    # multiplied later by scale
    scale = 1
    param_key, param_grid_args = list(params_vary[key][section_key].items())[0]
    param_grid = get_grid_from_list(param_grid_args)
    if type(param_grid) in [list, tuple]:
        scale = param_grid[1]
        param_grid = param_grid[0]
    
    save_path = f'{os.path.dirname(save_path)}/{section_key}_{param_key}/'
    # Iterate over grid
    for param_value in param_grid:
        # Fromat save path
        if isinstance(param_value, np.int64):
            save_folder = '{}{}_{}/'.format(save_path, param_key, param_value)
        else:
            save_folder = '{}{}_{:.1f}/'.format(save_path, param_key, param_value)
            
        # Scale laser parameter
        default_params[key][section_key][param_key] = param_value * scale
        if key == 'lasers':
            default_params[key][section_key]['E0'] = float(W_to_E0(default_params[key][section_key]))

        # Extract needed parameters for simulation script
        laser_params = [default_params['lasers'][key] for key in default_params['lasers'].keys()]
        simbox_params = default_params['simbox_params']
        
        # Simulation
        run_simulation_postprocess(laser_params, save_folder, simbox_params,
                                   geometry, low_memory_mode, n_threads,
                                   pol_idx, eps)
    print('Grid simulation finished!')
    
    
    
    
    

