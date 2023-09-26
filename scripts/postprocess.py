'''
Utility functions for postprocessing vacem simulation results:
- Calculate signal, background and discernible photons  

Author: Maksim Valialshchikov, @maxbalrog (github)
'''

import numpy as np
from scipy.constants import c
from scipy.constants import physical_constants

__all__ = ['lam_to_omega_in_eV', 'transform_spherical', 'beam_divergence']

hbar_eV, _, _ = physical_constants['natural unit of action in eV s']
eV_to_J, _, _ = physical_constants['electron volt']


def lam_to_omega_in_eV(lam):
    omega = 2*np.pi*c/lam * hbar_eV
    return omega


def transform_spherical(theta, phi, coll_angle, geometry='xz', eps=1e-10):
    '''
    Transformation between two spherical coordinate systems. For laser pulses
    propagating in 'xy' or 'xz' plane formulas are different
    
    theta: [np.array] - theta angle in original coordinate system
    phi: [np.array] - phi angle in original coordinate system
    coll_angle: collision angle [float] - angle between k vector
                and z axis (theta) for 'xz' collision and angle between
                k vector and x axis (phi) for 'xy' collision
    geometry: ['xy' or 'xz'] - collision geometry
    eps: [float] - for stability issues (dividing by zero)
    
    L stands for Lab (transformed) coordinate system
    '''
    coll_angle /= 180/np.pi
    if geometry == 'xy':
        cos_theta_L = np.cos(phi)*np.sin(theta)*np.cos(coll_angle) +\
                      np.sin(phi)*np.sin(theta)*np.sin(coll_angle)
        theta_L = np.arccos(cos_theta_L)
        phi_L = np.arccos(-np.cos(theta)/(np.sin(theta_L) + eps))
    elif geometry == 'xz':
        cos_theta_L = np.cos(phi)*np.sin(theta)*np.sin(coll_angle) +\
                      np.cos(theta)*np.cos(coll_angle)
        theta_L = np.arccos(cos_L)
        cos_phi_L = np.cos(phi)*np.sin(theta)*np.cos(coll_angle) -\
                    np.cos(theta)*np.sin(coll_angle)
        cos_phi_L = cos_phi_L / (np.sin(theta_L) + eps)
        phi_L = np.arccos(cos_phi_L)
    return theta_L, phi_L
    

def beam_divergence(theta, phi, laser_params, geometry):
    '''
    Calculate photon density for laser beam 
    (analytic expression from https://arxiv.org/abs/2205.15684).
    
    geometry: ['xy' or 'xz'] - collision geometry
    '''
    # Calculate angles in transformed coordinate frame
    coll_angle = laser_params['theta'] if geometry == 'xz' else laser_params['phi']
    theta_L, phi_L = transform_spherical(theta, phi, coll_angle, geometry)
    
    # Wavelength and frequency
    lam = laser_params['lam']
    omega_eV = lam_to_omega_in_eV(lam)
    omega0 = 2*np.pi/lam
    
    # Normalized number of photons
    N = W / (omega_eV * eV_to_J)
    N0 = N / (2*np.pi) * (omega0*w0)**2
    
    # Photon density of background
    exp = 0.5*(omega0*w0)**2*(mu1**2*np.cos(phi_L)**2 + mu2**2*np.sin(phi_L)**2)*theta_L**2
    Nbg = mu1 * mu2 * N0 * np.exp(-exp)
    return Nbg
    
    
    





