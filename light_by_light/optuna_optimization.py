'''
Optuna optimization functions for light-by-light scattering
with vacem simulations

Author: Maksim Valialshchikov, @maxbalrog (github)
'''
import numpy as np
import optuna
from copy import deepcopy
import os
from pathlib import Path
from functools import partial

from light_by_light.vacem_ini import W_to_E0
from light_by_light.vacem_simulation import run_simulation_postprocess
from light_by_light.utils import read_yaml

__all__ = ['get_trial_params', 'objective_lbl', 'run_optuna_optimization']


def get_trial_params(trial, optuna_params, default_params):
    '''
    Pick parameters for trial according to their data type and range
    '''
    params_upd = deepcopy(default_params)
    for laser_key in optuna_params.keys():
        for param_key in optuna_params[laser_key].keys():
            param = optuna_params[laser_key][param_key]
            param_name = f'{laser_key}/{param_key}'
            if param[-1] == 'int':
                step = param[2] if len(param) > 3 else None
                value = trial.suggest_int(param_name, param[0], param[1],
                                          step=step)
            elif param[-1] == 'float':
                step = param[2] if len(param) > 3 else None
                value = trial.suggest_float(param_name, param[0], param[1],
                                            step=step)
            elif param[-1] == 'uniform':
                value = trial.suggest_uniform(param_name, param[0], param[1])
            elif param[-1] == 'loguniform':
                value = trial.suggest_float(param_name, param[0], param[1], log=True)
            params_upd[laser_key][param_key] = value
        params_upd[laser_key]['E0'] = float(W_to_E0(params_upd[laser_key]))
    return params_upd


def objective_lbl(trial, default_params, optuna_params, save_path, geometry='xz',
                  obj_param='N_total', low_memory_mode=False, n_threads=12,
                  pol_idx=0, eps=1e-10):
    '''
    Objective function for optuna optimization.
    
    trial: [optuna.trial.Trial] - trial object
    default_params: [dict] - parameters for simulation including laser and
                             simbox parameters
    optuna_params: [dict] - parameters to optimize, each parameter should contrain
                            [start, end, (step), sampling]. Sampling should be one 
                            of ['int', 'float', 'uniform', 'loguniform']
    save_path: [str] - directory to save simulation results
    obj_param: ['N_total' or 'N_disc'] - quantity to maximize
    For other parameters check `run_simulation_postprocess()` 
    '''
    simbox_params = default_params['simbox_params']
    # Pick params for the trial
    params_upd = get_trial_params(trial, optuna_params, default_params['lasers'])
    
    # Save path formatting
    trial_str = str(trial.number).zfill(3)
    save_folder = f'{os.path.dirname(save_path)}/trials/trial_{trial_str}/'
    
    # Params for simulation
    laser_params = [params_upd[key] for key in params_upd.keys()]
    
    # Simulation 
    run_simulation_postprocess(laser_params, save_folder, simbox_params,
                               geometry, low_memory_mode, n_threads,
                               pol_idx, eps)
    
    # Extracting objective function
    result = np.load(f'{os.path.dirname(save_folder)}/postprocess_data.npz')
    for key in ["N_total", "Nperp_total", "N_disc", "Nperp_disc"]:
        trial.set_user_attr(key, float(result[key]))
    return float(result[obj_param])
    

def run_optuna_optimization(default_yaml, optuna_yaml, save_path, geometry='xz',
                            obj_param='N_total', n_trials=20,
                            low_memory_mode=False, n_threads=12,
                            pol_idx=0, eps=1e-10):
    # Read yaml files
    default_params = read_yaml(default_yaml)
    optuna_params = read_yaml(optuna_yaml)
    
    # String formatting for database
    study_name = save_path.split('/')[-2]
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    storage_name = f'sqlite:///{save_path}/study.db'

    # Create optuna study
    study = optuna.create_study(direction="maximize", study_name=study_name,
                                storage=storage_name, load_if_exists=True)

    # Pass necessary arguments to objective function
    obj = partial(objective_lbl,
                  default_params=default_params,
                  optuna_params=optuna_params,
                  save_path=save_path,
                  geometry=geometry,
                  obj_param=obj_param,
                  low_memory_mode=low_memory_mode,
                  n_threads=n_threads,
                  pol_idx=pol_idx)
    
    # Let the optimization begin!
    study.optimize(obj, n_trials=n_trials)
    print('Optuna optimization finished!')





