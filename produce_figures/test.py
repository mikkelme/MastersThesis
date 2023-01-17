import sys
sys.path.append('../') # parent folder: MastersThesis

from config_builder.build_config import *
from plot_set import *
import matplotlib.pyplot as plt
import numpy as np
from ase.visualize.plot import plot_atoms

# https://gitlab.com/ase/ase/blob/master/ase/visualize/plot.py


def show_coordinates(shape = (6,6)):
    # Build sheet 
    mat = np.ones((shape[0], shape[1])).astype('int')
    builder = config_builder(mat)
    Cdis = 1.461
    a = 3*Cdis/np.sqrt(3)
    
    # Plot patches
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    ax = plot_atoms(builder.sheet, radii = 0.8, show_unit_cell = 0, scale = 1, offset = (0,0))

    # Get min x and y for alligbment 
    xp, yp = [], [] # patches position
    for h in ax.patches:
        x, y = h.get_center()
        xp.append(x)
        yp.append(y)

    xs, ys = min(xp), min(yp) # xstart, ystart 

    # Plot coordinates on atom patches
    positions = builder.sheet.positions
    for i in range(shape[0]):
        verline = []
        for j in range(shape[1]):
            x, y = positions[i*shape[1]+j][:2] + (xs, ys)
            verline.append([x,y])
            plt.text(x, y, f'{i},{j}', horizontalalignment='center', 
                                       verticalalignment='center')

        verline = np.array(verline)
        plt.plot(verline[:,0], verline[:,1], linewidth = 0.5, linestyle = 'dashed', alpha = 0.75, color = 'grey', zorder = 0)
 
    
    
    # Change axis ticks to match coordinate system
    xtick_loc = np.array([xs + 1/2*a/(2*np.sqrt(3)) + i*a*np.sqrt(3)/2 for i in range(shape[0])])
    ytick_loc = np.array([ys + j*a*1/2 for j in range(shape[1])])
    ax.set_xticks(xtick_loc)
    ax.set_xticklabels(np.arange(len(xtick_loc)))
    ax.set_yticks(ytick_loc)
    ax.set_yticklabels(np.arange(len(ytick_loc)))
    
    ax.yaxis.grid(True) # horizontal lines
    ax.xaxis.grid(False) # vertical lines
    ax.set_axisbelow(True)
    
    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel(r"$y$", fontsize=14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.savefig("../article/figures/atom_indexing.pdf", bbox_inches="tight")



def show_center_coordinates(shape = (6,6)):
    # Build sheet 
    mat = np.ones((shape[0], shape[1])).astype('int')
    builder = config_builder(mat)
    Cdis = 1.461
    a = 3*Cdis/np.sqrt(3)
    
    # Plot patches
    plt.figure(num=1, dpi=80, facecolor='w', edgecolor='k')
    ax = plot_atoms(builder.sheet, radii = 0.8, show_unit_cell = 0, scale = 1, offset = (0,0))

    # Get min x and y for alligbment 
    xp, yp = [], [] # patches position
    for h in ax.patches:
        x, y = h.get_center()
        xp.append(x)
        yp.append(y)

    xs, ys = min(xp), min(yp) # xstart, ystart 


    # Plot center elements
    Lx = ((a*np.sqrt(3)/2) + a/(2*np.sqrt(3)))/2 
    for j in range(0, shape[1], 2):
        horline = []
        for i in range(0, shape[0]+1):
            x = xs + (1+3/2*i - 3/2) * Lx
            y = ys + a*(1/2 + 1/2*j - 1/2*(i%2)) 
            
            horline.append([x,y])
            circle = plt.Circle((x, y), 0.2,  color = color_cycle(9))
            ax.add_patch(circle)

        horline = np.array(horline)
        plt.plot(horline[:,0], horline[:,1], linewidth = 0.5, linestyle = 'dashed', alpha = 0.75, color = 'grey', zorder = 0)
 

    # Change axis ticks to match center coordinate system
    xtick_loc = [xs + (3*i/2+1) * Lx for i in range(-1, shape[0])]
    ytick_loc = [ys + 1/2*a*j for j in range(shape[1])]
    ax.set_xticks(xtick_loc)
    ax.set_xticklabels(np.arange(len(xtick_loc)))
    ax.set_yticks(ytick_loc)
    ax.set_yticklabels(np.arange(len(ytick_loc)))
    
    ax.yaxis.grid(False) # horizontal lines
    ax.xaxis.grid(True) # vertical lines
    ax.set_axisbelow(True)

    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel(r"$y$", fontsize=14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    plt.savefig("../article/figures/center_indexing.pdf", bbox_inches="tight")

if __name__ == "__main__":
    shape = (6, 6)
    show_coordinates(shape)
    show_center_coordinates(shape)
    plt.show()
    