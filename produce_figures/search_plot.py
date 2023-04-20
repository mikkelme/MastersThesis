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
    valid  = ~np.isnan(data[keys[0]])
    unique = np.argwhere(~np.isnan(data[keys[0]]))
    x_max = np.max(unique[:, 0])
    y_max = np.max(unique[:, 1])


    cmap = 'viridis'
    for i, key in enumerate(keys):
        plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
        ax = plt.gca()
        
        score = data[key][:x_max+1, :y_max+1]
        std = np.std(data[key][valid])
        rel_std = std/np.mean(data[key][valid])
        print(f'{key}: rel.std = {rel_std:0.4f}, std = {std:0.4f}')
        
        sns.heatmap(score, cmap = cmap, cbar_kws={'label': zlabel[i]}, annot = True, ax=ax)
        ax.invert_yaxis()
        ax.set_xlabel(r"$x$ (armchair direction)", fontsize = 14)
        ax.set_ylabel(r"$y$ (zigzag direction)", fontsize = 14)
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        
        if save:    
            savename = filename.strip('.npy').split('/')[-1]
            save_tag = keys[i].split('_')[-1]
            plt.savefig(f'../article/figures/search/ref_search_{save_tag}_{savename}.pdf', bbox_inches='tight')
        
    plt.show()
    


if __name__ == '__main__':
    # plot_ref_search(filename = '../ML/ref_search/pop_5_3_1_ref_search.npy', save = True)
    plot_ref_search(filename = '../ML/ref_search/hon_2_3_3_3_ref_search.npy', save = True)
    
    # plot_ref_search(filename = '../ML/ref_search/hon_2_1_1_1_ref_search.npy')
    # plot_ref_search(filename = '../ML/ref_search/hon_2_1_5_3_ref_search.npy')
    pass