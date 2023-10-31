'''
General utility functions

Author: Maksim Valialshchikov, @maxbalrog (github)
Original author of 'field_to_spherical': Alexander Blinne
'''
import yaml
import numpy as np
import postpic as pp

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
    for i,trial in enumerate(trials):
        for key in param_keys:
            data[key][i] = trial.params[key]
        for key in user_attr_keys:
            data[key][i] = trial.user_attrs[key]
    return data


def spherical_to_kartesian(r, theta, phi):
    """
    Converts a vector from spherical to kartesian coordinates

    Arguments:
    r, theta, phi: Components of vector in spherical coordinates
    """
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x, y, z


def cylindrical_to_kartesian(r, phi, z):
    """
    Converts a vector from cylindrical to kartesian coordinates

    Arguments:
    r, phi, z: Components of vector in cylindrical coordinates
    """
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    z = z
    return x, y, z


def field_to_spherical(field, phi_offset=0.0, match_resolution_radius=None,
                       preserve_integral=False, logscale=False,
                       angular_resolution=None, **kwargs):
    cval = 0
    if logscale:
        field = np.log(field)
        cval = -100
    kx, ky, kz = field.meshgrid()
    # kmax = np.sqrt(np.max(kx**2)+np.max(ky**2)+np.max(kz**2))
    kmax = np.max([np.max(np.abs(kx)), np.max(np.abs(ky)), np.max(np.abs(kz))])
    if match_resolution_radius is None:
        match_resolution_radius = kmax/3

    dk = np.min(field.spacing)
    nk = int(np.ceil(kmax/dk))
    if angular_resolution:
        dphi = dtheta = angular_resolution
    else:
        dphi = dtheta = dk/match_resolution_radius
    ntheta = int(round(np.pi/dtheta))
    nphi = int(round(2*np.pi/dphi))

    kAx = pp.Axis(name='k', grid=np.arange(0, dk*(nk+1), dk))
    thetaAx = pp.Axis(name='theta', grid=np.linspace(0, np.pi, ntheta))
    phiAx = pp.Axis(name='phi', grid=np.linspace(phi_offset, phi_offset+2*np.pi, nphi))
    field_spherical = field.map_coordinates([kAx, thetaAx, phiAx],
                                            transform = spherical_to_kartesian,
                                            preserve_integral=preserve_integral,
                                            cval=cval, **kwargs)
    if logscale:
        field_spherical = np.exp(field_spherical)
    field_spherical.unit=''
    return field_spherical
 

def field_to_cylindrical(field, phi_offset=0.0, match_resolution_radius=None,
                         preserve_integral=False, logscale=False,
                         angular_resolution=None, **kwargs):
    cval = 0
    if logscale:
        field = np.log(field)
        cval = -100
    kx, ky, kz = field.meshgrid()
    kmax = np.max([np.max(np.abs(kx)), np.max(np.abs(ky))])
    if match_resolution_radius is None:
        match_resolution_radius = kmax/3

    dk = np.min(field.spacing[:2])
    nk = int(np.ceil(kmax/dk))
    if angular_resolution:
        dphi = angular_resolution
    else:
        dphi = dk/match_resolution_radius
    nphi = int(round(2*np.pi/dphi))

    kAx = pp.Axis(name='k', grid=np.arange(0, dk*(nk+1), dk))
    phiAx = pp.Axis(name='phi', grid=np.linspace(phi_offset, phi_offset+2*np.pi, nphi))
    kzAx = field.axes[2]
    field_transformed = field.map_coordinates([kAx, phiAx, kzAx],
                                              transform = cylindrical_to_kartesian,
                                              preserve_integral=preserve_integral,
                                              cval=cval, **kwargs)
    if logscale:
        field_transformed = np.exp(field_transformed)
    field_transformed.unit=''
    return field_transformed
