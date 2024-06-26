'''
Optuna optimization functions for light-by-light scattering
with vacem simulations

Author: Maksim Valialshchikov, @maxbalrog (github)
'''
import numpy as np
from copy import deepcopy
import os
from pathlib import Path
from functools import partial
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from optuna.samplers import TPESampler

from light_by_light.vacem_ini import W_to_E0
from light_by_light.vacem_simulation import run_simulation_postprocess
from light_by_light.utils import read_yaml

__all__ = ['get_trial_params', 'objective_lbl', 'run_optuna_optimization']


def get_dependent_params(trial, total_value, params, param_key):
    '''
    If we have fixed energy budget for N pulses and want to optimize
    the energy distribution between them, then the energies of two pulses
    are dependent parameters.
    '''
    n_lasers = len(list(params.keys()))
    lasers = [f'laser_{idx}' for idx in range(n_lasers)]
    for laser_key in lasers[:n_lasers-1]:
        param_name = f'{laser_key}/{param_key}'
        value = trial.suggest_float(param_name, 0., total_value)
        params[laser_key][param_key] = float(value)
        params[laser_key]['E0'] = float(W_to_E0(params[laser_key]))
        total_value -= float(value)
    params[lasers[-1]][param_key] = float(total_value)
    params[lasers[-1]]['E0'] = float(W_to_E0(params[lasers[-1]]))
    trial.set_user_attr(f'{lasers[-1]}/{param_key}', float(total_value))
    return params


def get_trial_params(trial, optuna_params, default_params):
    '''
    Pick parameters for trial according to their data type and range.
    Param should contain following info:
    [start, end, (scale), sampling_type]
    '''
    params_upd = deepcopy(default_params)
    for laser_key in optuna_params.keys():
        for param_key in optuna_params[laser_key].keys():
            if param_key == 'W':
                W_total = float(optuna_params[laser_key][param_key][1])
                param_upd = get_dependent_params(trial, W_total, params_upd,
                                                 param_key)
            else:
                param = optuna_params[laser_key][param_key]
                param_name = f'{laser_key}/{param_key}'
                scale = 1 if len(param) == 3 else param[2]
                if param[-1] == 'int':
                    value = trial.suggest_int(param_name, param[0], param[1])
                elif param[-1] in ['float', 'uniform']:
                    value = trial.suggest_float(param_name, param[0], param[1])
                # elif param[-1] == 'uniform':
                #     value = trial.suggest_float(param_name, param[0], param[1])
                elif param[-1] == 'loguniform':
                    value = trial.suggest_float(param_name, param[0], param[1], log=True)
                params_upd[laser_key][param_key] = float(value * scale)
        params_upd[laser_key]['E0'] = float(W_to_E0(params_upd[laser_key]))
    return params_upd


def objective_lbl(trial, default_params, optuna_params, save_path, geometry='xz',
                  obj_param='N_total', low_memory_mode=False, n_threads=12,
                  pol_idx=0, eps=1e-10, discernible_spectral=False,
                  sphmap_params={'order': 1}, sph_limits=None):
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
                               pol_idx, eps,
                               discernible_spectral=discernible_spectral,
                               sphmap_params=sphmap_params, sph_limits=sph_limits)
    
    # Extracting objective function
    result = np.load(f'{os.path.dirname(save_folder)}/postprocess_data.npz')
    for key in ["N_total", "Nperp_total", "N_disc", "Nperp_disc",
                "N_disc_num", "Nperp_disc_num"]:
        if key in result.keys():
            trial.set_user_attr(key, float(result[key]))
    return float(result[obj_param])
    

def run_optuna_optimization(default_yaml, optuna_yaml, save_path, eps=1e-10,
                            SEED=2323):
    # Read yaml files
    default_params = read_yaml(default_yaml)
    optuna_params = read_yaml(optuna_yaml)
    
    # Define simulation parameters
    geometry = default_params['geometry']
    low_memory_mode = default_params['low_memory_mode']
    n_threads = default_params['n_threads']
    pol_idx = default_params['pol_idx']
    
    obj_param = default_params['obj_param']
    n_trials = default_params['n_trials']
    consider_endpoints = default_params.get('consider_endpoints', False)
    consider_prior = default_params.get('consider_prior', True)
    n_startup_trials = default_params.get('n_startup_trials', 10)
    
    sphmap_params = default_params.get('sphmap_params', {'order': 1})
    discernible_spectral = default_params.get('discernible_spectral', False)
    sph_limits = default_params.get('sph_limits', None)
    
    # String formatting for database
    study_name = save_path.split('/')[-2]
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    
    storage_name = f'{os.path.dirname(save_path)}/study.log'
    file_storage = optuna.storages.JournalFileStorage(storage_name)
    storage = optuna.storages.JournalStorage(file_storage)

    # Create optuna study or load if it already exists
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except:
        sampler = TPESampler(consider_prior=consider_prior, seed=SEED,
                             multivariate=True, n_startup_trials=n_startup_trials,
                             consider_endpoints=consider_endpoints)
        study = optuna.create_study(direction="maximize", study_name=study_name,
                                    storage=storage, load_if_exists=True,
                                    sampler=sampler)

    # Pass necessary arguments to objective function
    obj = partial(objective_lbl,
                  default_params=default_params,
                  optuna_params=optuna_params,
                  save_path=save_path,
                  geometry=geometry,
                  obj_param=obj_param,
                  low_memory_mode=low_memory_mode,
                  n_threads=n_threads,
                  pol_idx=pol_idx,
                  discernible_spectral=discernible_spectral,
                  sphmap_params=sphmap_params,
                  sph_limits=sph_limits)
    
    # Let the optimization begin!
    study.optimize(obj, n_trials=n_trials)
    print('Optuna optimization finished!')





