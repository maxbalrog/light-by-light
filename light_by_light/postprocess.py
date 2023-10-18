'''
Utility functions for postprocessing vacem simulation results:
- Calculate signal, background and discernible photons  

Author: Maksim Valialshchikov, @maxbalrog (github)
'''
import numpy as np
from scipy.constants import c
from scipy.constants import physical_constants
from pathlib import Path
import os

from vacem.support.resultfile import ResultFile
# from vacem.support.eval_functions import field_to_spherical

from light_by_light.utils import field_to_spherical
from light_by_light.laser_field import LaserBG

__all__ = ['lam_to_omega_in_eV', 'transform_spherical', 'beam_divergence',
           'SignalAnalyzer']

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
    elif geometry == 'xz' or geometry == 'z':
        cos_theta_L = np.cos(phi)*np.sin(theta)*np.sin(coll_angle) +\
                      np.cos(theta)*np.cos(coll_angle)
        theta_L = np.arccos(cos_theta_L)
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
    geometry_z = (geometry == 'xz') or (geometry == 'z')
    coll_angle = laser_params['theta'] if geometry_z else laser_params['phi']
    theta_L, phi_L = transform_spherical(theta[:,None], phi[None,:], coll_angle, geometry)
    
    # Necessary variables
    lam = laser_params['lam']
    omega_eV = lam_to_omega_in_eV(lam)
    omega0 = 2*np.pi/lam
    W = laser_params['W']
    if 'w0' in laser_params.keys():
        w0 = laser_params['w0']
        mu1 = mu2 = 1
    elif 'w0x' in laser_params.keys():
        w0 = laser_params['lam']
        mu1 = laser_params['w0x'] / w0
        mu2 = laser_params['w0y'] / w0
    
    # Normalized number of photons
    N = W / (omega_eV * eV_to_J)
    N0 = N / (2*np.pi) * (omega0*w0)**2
    
    # Photon density of background
    exp = 0.5*(omega0*w0)**2*(mu1**2*np.cos(phi_L)**2 + mu2**2*np.sin(phi_L)**2)*theta_L**2
    Nbg = mu1 * mu2 * N0 * np.exp(-exp)
    return Nbg


class SignalAnalyzer:
    def __init__(self, file, laser_pol, laser_params, geometry='xz'):
        '''
        Class to calculate signal photons (total and perp) from simulation results.
        Spherical density is calculated after initialization, discernible signal
        - with self.get_discernible_signal()
        
        file: [str] - path to vacem simulation results
        laser_pol: [instance of vacem.support.eval_functions.polarization_vector] - 
                    laser polarization vector
        laser_params: [list of dict] - parameters of laser in the collision
        geometry: ['xy' or 'xz'] - collision geometry
        '''
        self.result = ResultFile(file)
        self.laser_pol = laser_pol
        self.laser_params = laser_params
        self.geometry = geometry
        
        self.N_xyz = self.result.get_total_number_spectrum()
        self.Nperp_xyz = self.result.get_number_spectrum_polarized_spherical(laser_pol)
        self.x, self.y, self.z = [np.array(ax) for ax in self.N_xyz.axes]
        
        self.get_spherical_density()
        
        self.check_photon_number()
        
        # Laser diagnostics for numerical background
        ini_file = f'{os.path.dirname(file)}/vacem.ini'
        self.laser_diagnostics = LaserBG(ini_file)
        self.energy_num = self.laser_diagnostics.energy()
        
    def get_spherical_density(self):
        N = field_to_spherical(self.N_xyz, preserve_integral=False, order=1)
        Nperp = field_to_spherical(self.Nperp_xyz, preserve_integral=False, order=1)
        self.N = N
        self.Nperp = Nperp
        
        self.k, self.theta, self.phi = [np.array(ax) for ax in self.N.axes]
        self.dk = self.k[1] - self.k[0]
        self.dtheta = self.theta[1] - self.theta[0]
        self.dphi = self.phi[1] - self.phi[0]
        
        k, theta, phi = N.meshgrid()
        self.N_angular = N.evaluate('k**2 * N').integrate(0).matrix
        self.Nperp_angular = Nperp.evaluate('k**2 * Nperp').integrate(0).matrix
    
    def integrate_spherical(self, arr, axis=['k','theta','phi']):
        if 'k' in axis:
            integrated = np.sum(arr*self.k[:,None,None]**2, axis=0) * self.dk
        if 'phi' in axis:
            integrated = np.sum(integrated, axis=1) * self.dphi
        if 'theta' in axis:
            integrated = np.sum(integrated*np.sin(self.theta)) * self.dtheta
        return integrated
     
    def check_photon_number(self):
        N_total_xyz = self.N_xyz.integrate().matrix
        N_total_sph = self.integrate_spherical(self.N.matrix,
                                               axis=['k', 'theta','phi'])
        if not np.isclose(N_total_xyz, N_total_sph, rtol=5e-3):
            print('Warning: total signal on cartesian and spherical grid differ more than 0.5%')
        # assert np.isclose(N_total_xyz, N_total_sph, rtol=1e-2), f'Total number of signal\
        # photons for decart and spherical system is not the same:\n N_xyz = {N_total_xyz}\n\
        # N_sph = {N_total_sph}'
        self.N_total = N_total_sph
        
        Nperp_total_xyz = self.Nperp_xyz.integrate().matrix
        Nperp_total_sph = self.integrate_spherical(self.Nperp.matrix,
                                                   axis=['k','theta','phi'])
        # assert np.isclose(Nperp_total_xyz, Nperp_total_sph, rtol=1e-2), f'Total number of\
        # perp signal photons for decart and spherical system is not the same:\n\
        # N_xyz = {N_total_xyz}\n N_sph = {N_total_sph}'
        self.Nperp_total = Nperp_total_sph
        
    def get_background(self):
        self.background = 0
        self.background_lasers = []
        for laser in self.laser_params:
            background_ = beam_divergence(self.theta, self.phi,
                                          laser, self.geometry)
            self.background += background_
            self.background_lasers.append(background_)
        return self.background
    
    def get_background_num(self):
        self.laser_diagnostics.photon_density()
        self.background_num = self.laser_diagnostics.dphoton_angular.matrix
        self.background_sph_num = self.laser_diagnostics.dphoton_spherical.matrix
        return self.background_num
    
    def get_discernible_area(self, Nbg=None):
        if Nbg is None:
            Nbg = self.get_background()
        
        discernible_area, discernible_area_perp = [np.zeros_like(Nbg).astype(int) for i in range(2)]

        for idx_theta in range(len(self.theta)):
            idx = self.N_angular[idx_theta] > Nbg[idx_theta]
            discernible_area[idx_theta,idx] = 1
            idx = self.Nperp_angular[idx_theta] > self.pol_purity*Nbg[idx_theta]
            discernible_area_perp[idx_theta,idx] = 1
        return discernible_area, discernible_area_perp

    def _get_discernible_signal(self, discernible_area, discernible_area_perp):      
        N_disc, Nperp_disc = 0, 0
        for i in range(len(self.theta)):
            for j in range(len(self.phi)):
                if discernible_area[i,j]:
                    N_disc += self.N_angular[i,j] * np.sin(self.theta[i])
                if discernible_area_perp[i,j]:
                    Nperp_disc += self.Nperp_angular[i,j] * np.sin(self.theta[i])
        N_disc = N_disc * self.dphi * self.dtheta
        Nperp_disc = Nperp_disc * self.dphi * self.dtheta
        return N_disc, Nperp_disc
    
    def get_discernible_signal(self, pol_purity=1e-10):
        # Background could be calculated either with analytical formula or 
        # numerically. _num stands for numerically calculated values. In future
        # one should choose one method and stick to it.
        self.pol_purity = pol_purity
        
        Nbg = self.get_background()
        Nbg_num = self.get_background_num()
        
        areas = self.get_discernible_area(Nbg)
        self.discernible_area, self.discernible_area_perp = areas
        areas_num = self.get_discernible_area(Nbg_num)
        self.discernible_area_num, self.discernible_area_perp_num = areas_num
        
        self.N_disc, self.Nperp_disc = self._get_discernible_signal(*areas)
        self.N_disc_num, self.Nperp_disc_num = self._get_discernible_signal(*areas_num)
    
    def save_data(self, save_path):
        Path(f'{os.path.dirname(save_path)}').mkdir(parents=True, exist_ok=True)
        file = f'{os.path.dirname(save_path)}/postprocess_data.npz'
        data = {
            # 'k': self.k,
            'theta': self.theta,
            'phi': self.phi,
            # 'N_sph': self.N,
            # 'Nperp_sph': self.Nperp,
            'N_angular': self.N_angular,
            'Nperp_angular': self.Nperp_angular,
            'background': self.background,
            'discernible_area': self.discernible_area,
            'discernible_area_perp': self.discernible_area_perp,
            'N_total': self.N_total,
            'Nperp_total': self.Nperp_total,
            'N_disc': self.N_disc,
            'Nperp_disc': self.Nperp_disc,
            # 'background_sph_num': self.background_sph_num,
            'background_num': self.background_num,
            'discernible_area_num': self.discernible_area_num,
            'discernible_area_perp_num': self.discernible_area_perp_num,
            'N_disc_num': self.N_disc_num,
            'Nperp_disc_num': self.Nperp_disc_num,
        }
        np.savez(file, **data)
    
    
    





