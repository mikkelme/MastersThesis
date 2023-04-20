import sys
sys.path.append('../') # parent folder: MastersThesis
import matplotlib.pyplot as plt
import numpy as np

from plot_set import *
# from ML.hypertuning import *
# from ML.ML_perf import *
# from analysis.analysis_utils import*
# from matplotlib.gridspec import GridSpec


def plot_ref_search(filename, save = False):

    data = np.load(filename, allow_pickle = True).item()
    keys = [ 'Ff_min', 'Ff_max', 'Ff_max_diff', 'Ff_max_drop']
    zlabel = [r'Min $F_{fric}$', r'Max $F_{fric}$', r'Max $\Delta F_{fric}$', r'Max drop']

    # Get boundaries
    unique = np.argwhere(~np.isnan(data[keys[0]]))
    x_max = np.max(unique[:, 0])
    y_max = np.max(unique[:, 1])


    cmap = 'viridis'
    for i, key in enumerate(keys):
        plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
        im = plt.imshow(data[key][:x_max+1, :y_max+1].T, origin = 'lower', cmap = cmap)
        plt.xlabel(r"$x$ (armchair direction)", fontsize = 14)
        plt.ylabel(r"$y$ (zigzag direction)", fontsize = 14)
        plt.grid(False)

        cbar = plt.colorbar(im)
        cbar.set_label(zlabel[i])
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        
    plt.show()
    
    
    
    # plt.savefig('../article/figures/figure.pdf', bbox_inches='tight')
    
    # # Identify unique ref positions
    # refs = data['refs']
    # map = np.zeros((refs.shape[0], refs.shape[1])).astype('int')
    # map[0,0] = 1
    
    # # for k in range(2*np.max((map.shape[0], map.shape[1]))):
    # for k in range(np.max((map.shape[0], map.shape[1]))):
    #     for j in range(k+1):
    #         i = k - j
            
    #         if i == 0 and j == 0: break
    #         if i >= refs.shape[0]: break
    #         if j >= refs.shape[1]: break
            
    #         repeated_val = 0
    #         for key in keys:
    #             repeated_val += np.any(data[key][i,j] == data[key][np.nonzero(map)])
                
    #         if repeated_val == 0:
    #             map[i,j] = 1

    # plt.imshow(map.T, origin = 'lower')
    # plt.show()
    # exit()

    # cmap = 'viridis'
    # for i, key in enumerate(keys):
    #     plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    #     im = plt.imshow(data[key].T, origin = 'lower', cmap = cmap)
    #     plt.xlabel(r"$x$ (armchair direction)", fontsize = 14)
    #     plt.ylabel(r"$y$ (zigzag direction)", fontsize = 14)
    #     plt.grid(False)

    #     cbar = plt.colorbar(im)
    #     cbar.set_label(zlabel[i])
    #     plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        
    # plt.show()
    # # plt.savefig('../article/figures/figure.pdf', bbox_inches='tight')
    


if __name__ == '__main__':
    plot_ref_search(filename = '../ML/ref_search/test.npy')
    pass