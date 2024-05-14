'''
Utility functions for plotting 

Author: Maksim Valialshchikov, @maxbalrog (github)
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import os

from light_by_light.laser_field import LaserBG

__all__ = ['save_fig', 'plot_mollweide', 'get_E', 'plot_fields_ts',
           'plot_center', 'plot_fields']


def save_fig(figname):
    Path(os.path.dirname(figname)).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{figname}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'{figname}.png', bbox_inches='tight')
    return 1
    

def plot_mollweide(fig, ax, phi, theta, data, cmap='coolwarm', scale=None):
    theta_ = theta - np.pi/2
    phi_ = phi - np.pi
    phi_mesh, theta_mesh = np.meshgrid(phi_, theta_)
    
    if scale == 'log':
        norm = colors.LogNorm(vmin=data.min()+1e-10, vmax=data.max())
    else:
        norm = None
    
    im = ax.pcolormesh(phi_mesh, theta_mesh, data, cmap=cmap,
                       shading='gouraud', rasterized=True, norm=norm)
    fig.colorbar(im, ax=ax, shrink=0.5)

    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    xtick_labels = np.linspace(60, 360, 5, endpoint=False, dtype=int)
    ax.xaxis.set_ticklabels(r'$%s^{\circ}$' %num for num in xtick_labels)
    ytick_labels = np.linspace(0, 180, 7, endpoint=True, dtype=int)
    ax.yaxis.set_ticklabels(r'$%s^{\circ}$' %num for num in ytick_labels)
    for item in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        item.set_fontsize(18)
    ax.grid()
    return ax


def get_E(laser_bg):
    E = np.array(laser_bg.Ereal)
    E = np.sqrt(np.sum(E**2, axis=0))
    return E


def plot_fields_ts(vacem_ini, ts, figname='test'):
    '''
    Create field plots for several time steps (ts):
        - E in spatial domain
        - Nbg in frequency domain
        - S (Poynting vector)
    Print maximal field amplitude at boundaries (?)
    '''

    # Create our figures
    nt = len(ts)
    figs = [plt.figure(figsize=(nt*7,10), layout='constrained') for i in range(3)]

    # Go through time steps and gradually fill the figures
    Emax, Eb = -1, 0
    for i,t in enumerate(ts):
        laser_bg = LaserBG(vacem_ini, time=t)
        Nbg, Nbg_sph = laser_bg.photon_density()

        # compare max field at focus and max field at boundary
        if np.isclose(t*1e15,0):
            Emax = np.max(get_E(laser_bg))
        elif Emax > 0:
            E = get_E(laser_bg)
            Eb = 0
            Eb = np.max([Eb, np.max(E[[0,-1]])])
            Eb = np.max([Eb, np.max(E[:,[0,-1]])])
            Eb = np.max([Eb, np.max(E[:,:,[0,-1]])])
            boundary_ratio = Eb / Emax * 100
            total_ratio = E.max() / Emax * 100
            print(f'Max field at boundary: {boundary_ratio:.3f} %')
            print(f'Max field :            {total_ratio:.3f} %')

        # Plot field quantities
        z_idx = 'center' if i == len(ts)//2 else 'border'
        modes = ['E', 'Nbg', 'S']
        for idx, mode in enumerate(modes):
            plt.figure(figs[idx])
            plot_fields(laser_bg, t*1e15, mode=mode, ax_grid=(2,nt,i+1), z_idx=z_idx)
    
    if figname:
        names = ['E_spatial', 'N_freq', 'S_spatial']
        for fig, name in zip(figs, names):
            plt.figure(fig)
            save_fig(f'{figname}_{name}')
            fig.show()
    print('Everything is saved and plotted!')


def plot_center(ax):
    ax.axvline(0, 0, 1, color='white', linestyle='--')
    ax.axhline(0, 0, 1, color='white', linestyle='--')


def plot_fields(laser_bg, t, mode='E', ax_grid=(1,2,1), z_idx='border'):
    '''
    Plot electric field in coordinate space for several time step
    and planes 'xz', 'xy' (laser pulse propagating along z)
    '''
    if mode in ['E', 'S']:
        F = np.array(laser_bg.Ereal) if mode == 'E' else np.array(laser_bg.poynting_vec())
        F = np.sqrt(np.sum(F**2, axis=0))
        x, y, z = [np.array(laser_bg.x), np.array(laser_bg.y), np.array(laser_bg.z)]
        labels = [f'${ax}\: [\\mu m]$' for ax in ['x', 'y', 'z']]
        ax_scale = 1e6
    elif mode == 'Nbg':
        Nbg, Nbg_sph = laser_bg.photon_density()
        F = Nbg
        x, y, z = [np.array(ax) for ax in Nbg.axes]
        labels = [f'$k_{ax}$' for ax in ['x', 'y', 'z']]
        ax_scale = 1
    else:
        raise NotImplementedError(f"mode should be in [E, Nbg, S] but you passed: {mode}")
        
    nx, ny, nz = F.shape

    nrows, ncols, idx = ax_grid
    y_idx = ny//2
    # xz plane
    ax = plt.subplot(nrows, ncols, idx)
    plt.pcolormesh(z*ax_scale, x*ax_scale, F[:,y_idx,:])
    plt.colorbar()
    plot_center(ax)
    plt.xlabel(labels[2], fontsize=18)
    plt.ylabel(labels[0], fontsize=18)
    ax_comment = f', y={y[y_idx]*1e6:.1f}\: \\mu m' if mode in ['E', 'S'] else ''
    plt.title(f'${mode}\: (t={t:.0f}\: fs{ax_comment})$', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # xy plane
    idx_ = idx + ncols if nrows*ncols > 2 else idx+1
    z_idx = 0 if z_idx == 'border' else nz//2
    ax = plt.subplot(nrows, ncols, idx_)
    plt.pcolormesh(x*ax_scale, y*ax_scale, F[:,:,z_idx])
    plt.colorbar()
    plot_center(ax)
    plt.xlabel(labels[0], fontsize=18)
    plt.ylabel(labels[1], fontsize=18)
    ax_comment = f', y={z[z_idx]*1e6:.1f}\: \\mu m' if mode in ['E', 'S'] else ''
    plt.title(f'${mode}\: (t={t:.0f}\: fs{ax_comment})$', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return 1




    



    



