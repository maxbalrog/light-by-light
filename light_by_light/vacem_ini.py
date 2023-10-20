'''
Utility functions for creating .ini files for vacem

Author: Maksim Valialshchikov, @maxbalrog (github)
'''
import numpy as np
import postpic as pp
from scipy.constants import c, epsilon_0, mu_0, hbar

from pathlib import Path
import os
import yaml

from light_by_light.utils import write_yaml

__all__ = ['template_ini', 'template_laser', 'template_laser_ell',
           'W_to_E0', 'kmax_grid', 'get_spatial_steps', 'get_t_steps',
           'get_minmax_params', 'create_geometry', 'create_ini_file']


template_ini = '''
[Setup]
N = {Nx}, {Ny}, {Nz}
L = {Lx}, {Ly}, {Lz}
lasers = {lasers}
low_memory_mode = {low_memory_mode}

[Run]
t_start = {t_start}
t_end = {t_end}
t_steps = {t_steps}
fieldmode = {fieldmode}
'''

template_laser = '''
[laser_{laser_n}]
type = {solution}

focus_x = {x_foc}, {y_foc}, {z_foc}
focus_t = {t_foc}

theta = {theta}
phi = {phi}
beta = {beta}

lambda = {lam}
w0 = {w0}

tau = {tau}
E0 = {E0}
phi0 = {phi0}
order = {order}
spectrum_method = {spectrum_method}
'''

template_laser_ell = '''
[laser_{laser_n}]
type = {solution}

focus_x = {x_foc}, {y_foc}, {z_foc}
focus_t = {t_foc}

theta = {theta}
phi = {phi}
beta = {beta}

lambda = {lam}
w0x = {w0x}
w0y = {w0y}

tau = {tau}
E0 = {E0}
phi0 = {phi0}
order = {order}
spectrum_method = {spectrum_method}
'''

def W_to_E0(laser_params):
    '''
    Calculate maximum laser amplitude from total energy.
    Laser parameters must include W, tau and w0.
    '''
    W = laser_params['W']
    tau = laser_params['tau']
    if 'w0' in laser_params.keys():
        w0x = w0y = laser_params['w0']
    elif 'w0x' in laser_params.keys():
        w0x = laser_params['w0x']
        w0y = laser_params['w0y']
    return np.sqrt(8*np.sqrt(2/np.pi)*W/(np.pi * tau * w0x * w0y)/(c * epsilon_0))


def kmax_grid(laser_params):
    '''
    Laser params must include lam, tau, w0, theta and phi
    theta, phi in degrees
    '''
    lam, tau = laser_params['lam'], laser_params['tau']
    theta, phi = laser_params['theta'], laser_params['phi']
    if 'w0' in laser_params.keys():
        w0 = laser_params['w0']
    elif 'w0x' in laser_params.keys():
        w0 = min(laser_params['w0x'], laser_params['w0y'])
    
    k = 2*np.pi/lam    
    theta /= 180/np.pi
    phi /= 180/np.pi
    
    ekx = np.sin(theta) * np.cos(phi)
    eky = np.sin(theta) * np.sin(phi)
    ekz = np.cos(theta)
    ek = np.array([ekx, eky, ekz])

    if np.sin(theta) == 0.0:
        phi = 0.0
    
    e1x = -np.sin(phi)
    e1y = np.cos(phi)
    e1z = 0.0
    e1 = np.array([e1x, e1y, e1z])

    e2x = np.cos(phi) * np.cos(theta)
    e2y = np.sin(phi) * np.cos(theta)
    e2z = -np.sin(theta)
    e2 = np.array([e2x, e2y, e2z])
        
    kbw_perp = 4/w0
    kbw_long = 8/(c*tau)
    
    k0 = ek * k
    kmax = np.abs(k0 + ek * kbw_long)
    
    for beta in np.linspace(0, 2*np.pi, 64, endpoint=False):
        kp = k0 + ek * kbw_long + kbw_perp * (np.sin(beta) * e1 + np.cos(beta) * e2)
        kmax = np.maximum(kmax, np.abs(kp))
    
    return kmax


def get_spatial_steps(lasers, L, grid_res=1, equal_resolution=False):
    '''
    Calculates necessary spatial resolution
    
    grid_res: [float] - controls the resolution
    '''
    kmax = np.zeros((3,))
    for laser_params in lasers:
        kmax = np.maximum(kmax, kmax_grid(laser_params))
    
    # test cigar conjecture
    if equal_resolution:
        kmax = np.max(kmax) * np.ones(3) 
    N = np.ceil(grid_res * L * 3 * kmax/np.pi).astype(int)
    N = [pp.helper.fftw_padsize(n) for n in N]
    return N


def get_t_steps(t_start, t_end, lam, grid_res=1):
    '''
    Calculates necessary temporal resolution
    
    grid_res: [float] - controls the resolution
    '''
    fmax = c/lam
    return int(np.ceil((t_end-t_start)*fmax*6*grid_res))


def get_minmax_params(laser_params):
    '''
    From a list of laser parameters determine necessary min/max
    values (tau, w0 and lam).
    '''
    tau_max, w0_max, lam_min = 0, 0, 1e10
    for laser in laser_params:
        tau_max = max(tau_max, laser['tau'])
        lam_min = min(lam_min, laser['lam'])
        if 'w0' in laser.keys():
            w0_max = max(w0_max, laser['w0'])
        elif 'w0x' in laser.keys():
            w0_max = max([w0_max, laser['w0x'], laser['w0y']])
    return tau_max, w0_max, lam_min


def create_geometry(tau, w0, factors, geometry, pick_largest_size=False,
                    collision_axis='z'):
    '''
    Given geometry (e.g., 'xz'), calculate longitudinal (long)
    and transversal (trans) extent of spatial simulation box
    '''
    axes = ['x', 'y', 'z']
    L = {key: 0 for key in axes}
    long_axes = list(geometry)
    trans_axes = list(set(axes).difference(long_axes))
    
    trans_size = factors['trans'] * w0
    long_size = factors['long'] * c * tau
    if pick_largest_size:
        long_size = max([long_size, trans_size])
    for ax in long_axes:
        L[ax] = long_size if ax == collision_axis else max([long_size, trans_size])
    for ax in trans_axes:
        L[ax] = trans_size
    return np.array(list(L.values()))


def create_ini_file(laser_params, save_path, simbox_params,
                    geometry='xz', low_memory_mode=False):
    '''
    Create .ini file for vacem given laser parameters
    
    laser_params: [list of dict] - parameters of all laser pulses
    save_path: [str] - directory where .ini and .yaml files would be stored
    simbox_params: [dict] - consists of two dictionaries
        box_factors: [dict] - factors determining the size of simulation box
                          e.g., {'long': 10, 'trans': 5, 't': 2}
        resolutions: [dict] - spatial and temporal resolutions, default is 1
                              e.g., {'spatial': 2, 't': 1}
    geometry: [str] - spatial geometry, combination of 'x', 'y', 'z', e.g., 'xz'
    low_memory_mode: [bool] - parameter for vacem
    '''
    # Check that folder exists
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    
    # Extract information from parameters of laser pulses
    n_lasers = len(laser_params)
    tau, w0, lam = get_minmax_params(laser_params)
    # Information about simulation box
    box_factors = simbox_params['box_factors']
    resolutions = simbox_params['resolutions']
    equal_resolution = simbox_params.get('equal_resolution', False)
    pick_largest_size = simbox_params.get('pick_largest_size', False)
    
    # Create list of lasers
    laser_def = dict(phi0=0.0, x_foc=0.0, y_foc=0.0, z_foc=0.0, t_foc=0.0,
                     order=0, spectrum_method='complex')
    laser_list = [
        dict(**laser_def, **laser) for laser in laser_params
    ]
    
    # Determine time grid
    t_start, t_end = -box_factors['t']*tau, box_factors['t']*tau
    t_steps = get_t_steps(t_start, t_end, lam, resolutions['t'])
    
    # Determine spatial grid
    L = create_geometry(tau, w0, box_factors, geometry, pick_largest_size)
    N = get_spatial_steps(laser_list, L/2, resolutions['spatial'], equal_resolution)
    
    # Save some simulation parameters to yaml
    laser_data = {f'laser_{i}': params for i,params in enumerate(laser_params)}
    data = {'lasers': laser_data, 'simbox_params': simbox_params}
    yaml_file = f'{os.path.dirname(save_path)}/laser_simbox_params.yml'
    write_yaml(yaml_file, data)
    
    # Create and save vacem.ini file
    template = template_ini.format(Nx=N[0], Ny=N[1], Nz=N[2],
                                   Lx=L[0], Ly=L[1], Lz=L[2],
                                   lasers=n_lasers, low_memory_mode=low_memory_mode,
                                   t_start=t_start, t_end=t_end, t_steps=t_steps,
                                   fieldmode='solver')
    
    for i,laser in enumerate(laser_list):
        template_pulse = template_laser if 'w0' in laser.keys() else template_laser_ell
        template += template_pulse.format(laser_n=i+1, **laser)
    
    with open(f'{save_path}/vacem.ini', 'w+') as f:
        f.write(template)
    return None

