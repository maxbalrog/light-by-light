'''
Utility functions for plotting 

Author: Maksim Valialshchikov, @maxbalrog (github)
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

__all__ = ['plot_mollweide']


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



