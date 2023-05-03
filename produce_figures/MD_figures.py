import sys
sys.path.append('../') # parent folder: MastersThesis
import matplotlib.pyplot as plt
import numpy as np
from plot_set import *


def LJ_plot(save = False):

    eps = 1
    sigma = 1
    r = np.linspace(0.9, 2, int(1e4)) 
    rep = 4*eps * (sigma/r)**12
    atr = -4*eps * (sigma/r)**6
    LJ = rep + atr
    
    plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(r, LJ, label = 'LJ potential', linewidth = 1.5,  color = 'black', zorder = 4)
    plt.plot(r, rep, '--', label = 'Repulsion', color =  color_cycle(1), zorder = 3)
    plt.plot(r, atr, '--', label = 'Attraction', color =  color_cycle(0), zorder = 2)
    plt.legend(fontsize = 13)
    plt.xlabel(r'$r/\sigma$', fontsize=14)
    plt.ylabel(r'$E(r)/\epsilon$', fontsize=14)
    
    # Annotations 
    hline(plt.gca(), 0, color = 'grey', linewidth = 1, zorder = 1)

    arrowprops = {'arrowstyle': '<->', 'color': 'black', 'lw': 1.5}
    plt.annotate(text = '', xy=(2**(1/6), 0), xytext=(2**(1/6), -1.0), arrowprops=arrowprops)
    plt.text(2**(1/6) + 0.05, -0.5, r'$\epsilon$', horizontalalignment = 'right', fontsize = 20)
    
    plt.plot(sigma, 0, 'o', markersize = 5, color= 'black')
    plt.text(sigma + 0.05, + 0.03, r'$\sigma$', horizontalalignment = 'right', fontsize = 20)
    
    
    plt.xlim([r[0], r[-1]])
    plt.ylim([-2.0, 2.0])

    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
        plt.savefig('../article/figures/theory/LJ_pot.pdf', bbox_inches='tight')

if __name__ == '__main__':
    LJ_plot(save = True)
    plt.show()