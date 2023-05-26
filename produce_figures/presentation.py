import sys
sys.path.append('../') # parent folder: MastersThesis
import matplotlib.pyplot as plt
import numpy as np
from plot_set import *


def prop_of_interest(save = False):
    x = np.linspace(0, 1, int(1e4))
    a1 = 2.7
    a2 = -6
    a3 = 3.5
    c = 0.075
    y = c + a1*x + a2*x**2 + a3*x**3 
    
    
    plt.figure(num=0, figsize = (7,4), facecolor='w', edgecolor='k')
    plt.plot(x,y, zorder = -1)
    plt.xlabel('Strain', fontsize=14)
    plt.ylabel(r'$F_{fric}$', fontsize=14)
    plt.ylim(bottom = 0)
    
    argmax = np.argmax(y)
    argmin = np.argmin(y)
    
    offset = len(x)//2
    argmin2 = np.argmin(y[offset:])
    
    # Annotate
    xoffset = 0.02
    arrowprops = {'arrowstyle': '<->', 'color': 'black', 'lw': 1.5}
    bbox = dict(facecolor='white', edgecolor = 'None',  alpha=0.8)
    
    # delta max
    plt.text(x[argmax] + xoffset, (y[argmax]+y[argmin])/2, r'$\max \ \Delta F_{fric}$', horizontalalignment = 'left', fontsize = 14)
    plt.annotate('', xy=(x[argmax], y[argmin]), xytext=(x[argmax], y[argmax]), textcoords='data', arrowprops=arrowprops)
    
    # max drop
    plt.text(x[offset + argmin2] + xoffset, (y[argmax]+y[offset + argmin2])/2, r'Max drop', horizontalalignment = 'left', fontsize = 14)
    plt.annotate('', xy=(x[offset + argmin2], y[offset + argmin2]), xytext=(x[offset + argmin2], y[argmax]), textcoords='data', arrowprops=arrowprops)
    
    
    # min fric
    plt.scatter(x[argmin], y[argmin], edgecolor = 'black', zorder = 2, color = color_cycle(1), label = r'$\min \ F_{fric}$')
    hline(plt.gca(), y[argmin], zorder = -1, color = color_cycle(1), linestyle = '--', linewidth = 1)
    
    # max fric
    plt.scatter(x[argmax], y[argmax], edgecolor = 'black', zorder = 2, color = color_cycle(2), label = r'$\max \ F_{fric}$')
    hline(plt.gca(), y[argmax], zorder = -1, color = color_cycle(2), linestyle = '--', linewidth = 1)
    
    
    # plt.legend(loc = 'lower center', fontsize = 13, ncols = 2)
    # plt.legend(loc = 'upper center', bbox_to_anchor = (0.05, -0.01, 1, 1), bbox_transform = plt.gcf().transFigure, ncols = 2, fontsize = 13)
    plt.legend(loc = 'right', bbox_to_anchor = (0.01, 0, 1, 1), bbox_transform = plt.gcf().transFigure, ncols = 1, fontsize = 13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.subplots_adjust(right=0.8)

    if save:
        plt.savefig('../presentation/figures/prop_of_interets.pdf', bbox_inches='tight')


if __name__ == "__main__":
    prop_of_interest(save = True)    
    plt.show()