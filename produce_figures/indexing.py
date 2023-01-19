import sys
sys.path.append('../') # parent folder: MastersThesis
import matplotlib.pyplot as plt
import numpy as np
from plot_set import *
from analysis.analysis_utils import *
from config_builder.build_config import *
from ase.visualize.plot import plot_atoms


# https://gitlab.com/ase/ase/blob/master/ase/visualize/plot.py


def common_settings(tight_layout = True):
    plt.xlabel(r"$x$ (armchair direction)", fontsize=14)
    plt.ylabel(r"$y$ (zigzag direction)" , fontsize=14)
    if tight_layout:
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    


def coordinates(shape = (6,6), save = False):
    # Build sheet 
    mat = np.ones((shape[0], shape[1])).astype('int')
    builder = config_builder(mat)
    Cdis = 1.461
    a = 3*Cdis/np.sqrt(3)
    atom_radii = 0.8
    
    
    # Plot patches
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    # plt.title("Atom indexing")
    ax = plot_atoms(builder.sheet, radii = atom_radii, show_unit_cell = 0, scale = 1, offset = (0,0))

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
            plt.text(x, y, f'({i},{j})', horizontalalignment='center', 
                                       verticalalignment='center')

        verline = np.array(verline)
        plt.plot(verline[:,0], verline[:,1], linewidth = 0.5, linestyle = 'dashed', alpha = 0.75, color = 'grey', zorder = 0)
 
    Bx = a/(np.sqrt(3)*2)
    vecax = a*np.sqrt(3)/2
    vecay = a/2
    
    # Change axis ticks to match coordinate system
    xtick_loc = np.array([xs  + ((i+1)//2)*(vecax+Bx) + ((i)//2)*2*Bx for i in range(shape[0])])
    ytick_loc = np.array([ys + j*vecay for j in range(shape[1])])
    ax.set_xticks(xtick_loc)
    ax.set_xticklabels(np.arange(len(xtick_loc)))
    ax.set_yticks(ytick_loc)
    ax.set_yticklabels(np.arange(len(ytick_loc)))
    
    ax.yaxis.grid(True) # horizontal lines
    ax.xaxis.grid(False) # vertical lines
    ax.set_axisbelow(True)
    
    plt.xlabel(r"$x$ (armchair direction)", fontsize=14)
    plt.ylabel(r"$y$ (zigzag direction)" , fontsize=14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    if save:
        plt.savefig("../article/figures/atom_indexing.pdf", bbox_inches="tight")


def center_coordinates(shape = (6,6), save = False):
    # Build sheet 
    mat = np.ones((shape[0], shape[1])).astype('int')
    builder = config_builder(mat)
    Cdis = 1.461
    a = 3*Cdis/np.sqrt(3)
    atom_radii = 0.8
    center_radii = 0.4
    
    # Plot patches
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    # plt.title("Center element indexing")
    ax = plot_atoms(builder.sheet, radii = atom_radii, show_unit_cell = 0, scale = 1, offset = (0,0))

    # Get min x and y for alligbment 
    xp, yp = [], [] # patches position
    for h in ax.patches:
        x, y = h.get_center()
        xp.append(x)
        yp.append(y)

    xs, ys = min(xp), min(yp) # xstart, ystart 


    Bx = a/(np.sqrt(3)*2)
    vecax = a*np.sqrt(3)/2
    vecay = a/2
    

    # Plot center elements and coordinates
    Lx = (vecax + Bx)/2 
    for j in range(0, shape[1], 2):
        horline = []
        for i in range(0, shape[0]+1):
            x = xs + (1+ 3/2*(i - 1)) * Lx
            y = ys + a*(1/2 + 1/2*j - 1/2*(i%2)) 
            
            plt.text(x, y, f'({i},{j//2})', horizontalalignment='center', 
                                       verticalalignment='center')
            
            horline.append([x,y])
            circle = plt.Circle((x, y), center_radii,  color = color_cycle(6))
            ax.add_patch(circle)

        horline = np.array(horline)
        plt.plot(horline[:,0], horline[:,1], linewidth = 0.5, linestyle = 'dashed', alpha = 0.75, color = 'grey', zorder = 0)
 

    # Change axis ticks to match center coordinate system
    xtick_loc = [xs + (3*i/2+1) * Lx for i in range(-1, shape[0])]
    ytick_loc = [ys + vecay*(2*j + 1) for j in range(shape[1]//2)]
    ax.set_xticks(xtick_loc)
    ax.set_xticklabels(np.arange(len(xtick_loc)))
    ax.set_yticks(ytick_loc)
    ax.set_yticklabels(np.arange(len(ytick_loc)))
    
    
    ax.set_xlim([xs + -Lx/2 - 3/2*center_radii, xs + (1+ 3/2*(shape[0] - 1)) * Lx + 3/2*center_radii])
    
    ax.yaxis.grid(False) # horizontal lines
    ax.xaxis.grid(True) # vertical lines
    ax.set_axisbelow(True)


    common_settings()
    if save:
        plt.savefig("../article/figures/center_indexing.pdf", bbox_inches="tight")


def center_directions(save = False):
     # Build sheet 
    shape = (2,6)
    mat = np.ones((shape[0], shape[1])).astype('int')
    builder = config_builder(mat)
    Cdis = 1.461
    a = 3*Cdis/np.sqrt(3)
    atom_radii = 0.8
    center_radii = 0.65
    
    rel_dict = {(1,1): '$(i,j)$',
                (1,2): '$(i,j+1)$', 
                (2,1): '$(i+1,j)$', 
                (2,0): '$(i+1,j-1)$', 
                (1,0): '$(i,j-1)$', 
                (0,0): '$(i-1,j-1)$', 
                (0,1): '$(i-1,j)$'}
        
    
    titles = ['$i$ is odd', '$i$ is even']
    fig = plt.figure(num = unique_fignum(), figsize = (10,5))
    for k in range(2):
        up = k
        ax = plt.subplot(1,2, 2-k)
        plt.title(titles[k])
        # Plot patches
        plot_atoms(builder.sheet, radii = atom_radii, show_unit_cell = 0, scale = 1, offset = (0,0))
        
        # Get min x and y for alligbment 
        xp, yp = [], [] # patches position
        for h in ax.patches:
            x, y = h.get_center()
            xp.append(x)
            yp.append(y)
        xs, ys = min(xp), min(yp) # xstart, ystart 

        Bx = a/(np.sqrt(3)*2)
        vecax = a*np.sqrt(3)/2
        vecay = a/2
        
        # Plot center elements and coordinates
        Lx = (vecax + Bx)/2 
        
        if up:
            x = [xs + (1+ 3/2*(i - 1)) * Lx for i in range(shape[0]+1)]
            y = [ys - vecay + vecay/2*((j+4)%4) for j in range(0, shape[1], 2)]
            plt.plot(x, y, linewidth = 0.5, linestyle = 'dashed', alpha = 0.75, color = 'grey', zorder = 0)
            
            
        
        if up:
            rel_dict[(2,1)] = '$(i+1,j+1)$'
            rel_dict[(2,0)] = '$(i+1,j)$'
            rel_dict[(0,0)] = '$(i-1,j)$'
            rel_dict[(0,1)] = '$(i-1,j+1)$'
            
        
        for j in range(0, shape[1], 2):
            horline = []
            for i in range(0, shape[0]+1):
                x = xs + (1+ 3/2*(i - 1)) * Lx
                y = ys + a*(1/2 + 1/2*j - 1/2*(i%2)) 
                
                if (i,j//2) in rel_dict:
                    plt.text(x, y, rel_dict[(i,j//2)], horizontalalignment='center', 
                                                verticalalignment='center')
                    
                circle = plt.Circle((x, y), center_radii,  color = color_cycle(6))
                ax.add_patch(circle)
                if up:
                    if i%2:
                        y += 2*vecay
                    
                horline.append([x,y])

            horline = np.array(horline)
            plt.plot(horline[:,0], horline[:,1], linewidth = 0.5, linestyle = 'dashed', alpha = 0.75, color = 'grey', zorder = 0)
    

        # Change axis ticks to match center coordinate system
        xtick_loc = [xs + (3*i/2+1) * Lx for i in range(-1, shape[0])]
        ytick_loc = [ys + vecay*(2*j + 1) for j in range(shape[1]//2)]
        ax.set_xticks(xtick_loc)
        ax.set_xticklabels(['$i-1$', '$i$', '$i+1$'])
        ax.set_yticks(ytick_loc)
        ax.set_yticklabels(['$j-1$', '$j$', '$j+1$'])
        
        
        ax.set_xlim([xs + -Lx/2 - 3/2*center_radii, xs + (1+ 3/2*(shape[0] - 1)) * Lx + 3/2*center_radii])
        
        ax.yaxis.grid(False) # horizontal lines
        ax.xaxis.grid(True) # vertical lines
        ax.set_axisbelow(True)
        common_settings(tight_layout = False)
            
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)

    if save:
        fig.savefig("../article/figures/center_directions.pdf", bbox_inches="tight")

    

if __name__ == "__main__":
    shape = (6, 6)
    coordinates(shape, True)
    center_coordinates(shape, True)
    center_directions(True)
    plt.show()
    