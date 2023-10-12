# -*- coding: utf-8 -*-
"""
Created on Fri May 26 10:59:27 2023

@author: fabian.schuetze
@modified: maksim.valialshchikov
"""

import numpy as np
import postpic as pp
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.constants import c, epsilon_0, mu_0, hbar

from vacem.fields.solver import FieldSolver
from vacem.fields.common import SolverBase
from vacem.fields.parse_config import ConfigParser
from vacem.fields.fieldinput import ComplexEInput
from vacem.fields.lasers import get_lasers_from_config, get_a12_from_sources
from vacem.support.eval_functions import field_to_spherical

from light_by_light.vacem_ini import W_to_E0


class LaserBG:
    def __init__(self, ini_file, time=0.):
        '''
        Class for numerical field background estimation
        '''
        
        self.ini_file = ini_file
        
        config = ConfigParser()
        config.read(ini_file)
        
        self.config = config
        
        self.sb = SolverBase.from_setup_section(config['Setup'])
        self.sb.c = c
        self.axes = self.sb.grid
        self.x, self.y, self.z = self.axes
        
        lasers = get_lasers_from_config(self.sb, self.config)
        
        a12 = get_a12_from_sources(lasers, time)
        self.a1, self.a2 = a12

        self.fs = FieldSolver(self.sb)
        self.fs.setup(*a12)
        
        self.time = time
    
    def energy(self):
        """
        Calculates the energy contained in the field from the Fourier
        coefficients.
        """
        a1 = self.a1
        a2 = self.a2
        kx, ky, kz = a1.meshgrid()
        
        prefactor = 0.5*epsilon_0*c**2
        
        nrg_dens = a1.evaluate('prefactor * (kx**2 + ky**2 + kz**2)' \
                   + '*(a1.real**2 + a1.imag**2 + a2.real**2 + a2.imag**2)')
        nrg = nrg_dens.integrate(method='fast')
        self.nrg = nrg.matrix
        
        return self.nrg
    
    def energy_from_eb(self):
        """
        Calculates the energy contained in the field from the E & B field in
        position space.
        """
        Eabs2 = np.sum([np.abs(f)**2 for f in self.Ereal], axis=0)
        Babs2 = np.sum([np.abs(f)**2 for f in self.Breal], axis=0)
        
        u = 1/2 * (epsilon_0*Eabs2 + 1/mu_0*Babs2) # energy density
        d = [x[1]-x[0] for x in self.sb.grid]
        dV = np.product(d)
        self.nrg_from_eb = u.sum() * dV # total energy
        
        return self.nrg_from_eb

    def photon_density(self, preserve_integral=False, fix_resolution_radius=False):
        """
        Calculates the differential photon number in momentum space.
        """
        a1 = self.a1
        a2 = self.a2
        kx, ky, kz = a1.meshgrid()
        
        prefactor = 0.5*epsilon_0*c/hbar
        
        # calculate photon density on cartesian grid
        dphoton = a1.evaluate('prefactor * sqrt(kx**2 + ky**2 + kz**2)' \
                      + '*(a1.real**2 + a1.imag**2 + a2.real**2 + a2.imag**2)')
            
        for ax, name in zip(dphoton.axes, 'xyz'):
            ax.name = 'k' + name
        self.dphoton = dphoton
        
        if fix_resolution_radius:
            resolution_radius = np.min([kx.max(), ky.max(), kz.max()])
        else:
            resolution_radius = None
        
        # map photon density to spherical grid
        dphoton_spherical = field_to_spherical(dphoton, preserve_integral=preserve_integral,
                                               match_resolution_radius=resolution_radius,
                                               order=1)
        k, theta, phi = dphoton_spherical.meshgrid()
        self.dphoton_spherical = dphoton_spherical
        # self.k, self.theta, self.phi = k, theta, phi
        self.k, self.theta, self.phi = dphoton_spherical.grid
        
        # integrate over energies
        dphoton_angular = dphoton_spherical.evaluate('k**2 * dphoton_spherical').integrate(0)
        self.dphoton_angular = dphoton_angular
        
        # total number of photons
        photon_xyz = dphoton.integrate().matrix
        dphoton_theta = dphoton_angular.integrate(1)
        theta = np.array(dphoton_theta.axes[0])
        photon_sph = dphoton_theta.evaluate('dphoton_theta * sin(theta)').integrate().matrix
        self.photon_xyz = photon_xyz
        self.photon_sph = photon_sph
        
        err = np.abs(photon_xyz - photon_sph) / photon_xyz
        if err > 1e-3:
            print("Warning: total signal on cartesian and spherical grid differ more than 0.1%")
        
        return self.dphoton, self.dphoton_spherical

    
    def poynting_vec(self):
        """
        Calculates the time averaged Poynting vector field.
        """
        Hx_conj, Hy_conj, Hz_conj = [np.conj(f)/mu_0 for f in self.B]
        Ex, Ey, Ez = self.E
        Sx = 0.5 * (Ey*Hz_conj - Ez*Hy_conj)
        Sy = 0.5 * (Ez*Hx_conj - Ex*Hz_conj)
        Sz = 0.5 * (Ex*Hy_conj - Ey*Hx_conj)
        
        self.Sx = Sx.real
        self.Sy = Sy.real
        self.Sz = Sz.real
        
        return self.Sx, self.Sy, self.Sz
           
    def _set_up_fs(self, t):
        """
        Settting up the FieldSolver.
        """
        self.fs.t_or_x0 = self.time
        self.sb.t_or_x0 = self.time
    
    def _calculate_eb(self, t):
        """
        Calculates the E & B field in position space.
        """
        self.E = [c*f for f in self.fs.eb[:3]]
        self.B = self.fs.eb[3:]
        
        self.Ereal = [f.real for f in self.E]
        self.Breal = [f.real for f in self.B]
    
    @property
    def time(self):
        """
        Getting time ...
        """
        return self._time
    
    @time.setter
    def time(self, t):
        """
        Setting time ...
        """
        self._time = t
        
        self._set_up_fs(t)
        self._calculate_eb(t)
    
    
class LaserBGExplicit:
    def __init__(self, ini_file, time=0., laser_params=None):
        
        self.ini_file = ini_file
        self.laser_params = laser_params
        
        config = ConfigParser()
        config.read(ini_file)
        
        self.config = config
        
        self.sb = SolverBase.from_setup_section(config['Setup'])
        self.sb.c = c
        self.axes = self.sb.grid
        self.ppax = self.sb.axes
        self.x, self.y, self.z = self.axes
        
        Ex_, Ey_, Ez_ = self.paraxial_gaussian(self.x,self.y,self.z,
                                               laser_params[0])
        Ex = pp.Field(Ex_, axes=self.ppax)
        
        self.fs = FieldSolver(self.sb)

        las = ComplexEInput(self.sb, focus_t=0.0, focus_x=(0, 0, 0), Ex=Ex, Ey=0, Ez=0,
                            theta=0, phi=0, beta=0)
        self.a1, self.a2 = las.get_a12(0.0)
        self.fs.setup(*(self.a1, self.a2))
        
    def get_photon_density(self):
        a1, a2 = self.a1, self.a2
        kx, ky, kz = a1.meshgrid()
        
        prefactor = 0.5*epsilon_0*c/hbar

        dphoton = a1.evaluate('prefactor * sqrt(kx**2 + ky**2 + kz**2)' \
                      + '*(a1.real**2 + a1.imag**2 + a2.real**2 + a2.imag**2)')

        for ax, name in zip(dphoton.axes, 'xyz'):
            ax.name = 'k' + name
        self.dphoton = dphoton

        dphoton_spherical = field_to_spherical(dphoton, preserve_integral=False)
        k, theta, phi = dphoton_spherical.meshgrid()
        photon_dens = dphoton_spherical.evaluate('k**2 * dphoton_spherical').integrate(0, method='fast')
        return dphoton, photon_dens
        
    @staticmethod
    def paraxial_gaussian(x, y, z, laser_params):
        x = x[:,None,None]
        y = y[None,:,None]
        z = z[None,None,:]

        E0 = W_to_E0(laser_params)
        w0 = laser_params['w0']
        lam = laser_params['lam']
        tau = laser_params['tau']
        phi0 = np.pi/2
        
        k = 2*np.pi/lam
        r2 = x**2 + y**2
        zr = np.pi*w0**2/lam
        wr = w0 * np.sqrt(1 + (z/zr)**2)
        Rz_inv = z/(z**2 + zr**2)
        psi = np.arctan(z/zr)
        phase = phi0 - k*z - 0.5*k*r2*Rz_inv + psi
        envelope = np.exp(-(z/c)**2/(tau/2)**2)
        Ex = E0 * w0/wr * np.exp(-r2/wr**2) * 1.0j*np.exp(-1.0j*phase) * envelope
        return Ex, 0, 0
    
    def _set_up_fs(self, t):
        """
        Settting up the FieldSolver.
        """
        self.fs.t_or_x0 = self.time
        self.sb.t_or_x0 = self.time
    
    def _calculate_eb(self, t):
        """
        Calculates the E & B field in position space.
        """
        self.E = [c*f for f in self.fs.eb[:3]]
        self.B = self.fs.eb[3:]
        
        self.Ereal = [f.real for f in self.E]
        self.Breal = [f.real for f in self.B] 
    
    @property
    def time(self):
        """
        Getting time ...
        """
        return self._time
    
    @time.setter
    def time(self, t):
        """
        Setting time ...
        """
        self._time = t
        
        self._set_up_fs(t)
        self._calculate_eb(t)