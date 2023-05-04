import sys
sys.path.append('../') # parent folder: MastersThesis
import matplotlib.pyplot as plt
import numpy as np

from plot_set import *
from kirigami_patterns import *
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
        
        sns.heatmap(score.T, cmap = cmap, cbar_kws={'label': zlabel[i]}, annot = True, ax=ax)
        ax.invert_yaxis()
        ax.set_xlabel(r"$x$ (armchair direction)", fontsize = 14)
        ax.set_ylabel(r"$y$ (zigzag direction)", fontsize = 14)
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        
        if save:    
            savename = filename.strip('.npy').split('/')[-1]
            save_tag = keys[i].split('_')[-1]
            plt.savefig(f'../article/figures/search/ref_search_{save_tag}_{savename}.pdf', bbox_inches='tight')
        
    plt.show()
    


def plot_RW_top5(save = False):
    path = '../ML/RW_search/'

    categories = ['Ff_min', 'Ff_max', 'Ff_max_diff', 'Ff_max_drop']
    ylabel = [r'Min $F_{fric}$', r'Max $F_{fric}$', r'Max $\Delta F_{fric}$', 'Max drop']                  
    fig, axes = plt.subplots(4, 5, num = unique_fignum(), figsize = (10, 8))    
    
    for i, cat in enumerate(categories):
        config_paths = [os.path.join(path, f'{cat}{k}_conf.npy') for k in range(5)]
        for j, config_path in enumerate(config_paths):
            print(i, j)
            ax = axes[i,j]
            if i == 0:
                ax.set_title(f'# {j+1}')
            if j == 0:
                ax.set_ylabel(ylabel[i])
            ax.set_facecolor("white")
            ax.set_xticks([])
            ax.set_yticks([])
            
            
            # if i != j:
            #     continue
            mat = np.load(config_path)
            
            plot_sheet(mat, ax, radius = 0.6, facecolor = 'black', edgecolor = 'black', linewidth = 0.1)
            ax.grid(False)
        
    fig.supxlabel(r"$x$ (armchair direction)", fontsize = 14)
    fig.supylabel(r"$y$ (zigzag direction)", fontsize = 14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    if save:
        fig.savefig('../article/figures/search/RW_search_top5.pdf', bbox_inches='tight')


def plot_search_history(filename, save = False):
    # read score history
    infile = open(filename, 'r')
    gen, min_score, mean_score, max_score = [], [], [], []
    for line in infile:
        words = line.split(',')
        gen.append(int(words[0]))
        min_score.append(float(words[1]))
        mean_score.append(float(words[2]))
        max_score.append(float(words[3]))
        if gen[-1] == 300:
            break
        
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    plt.plot(gen, max_score, label = 'Maximum', color = color_cycle(1))
    plt.plot(gen, mean_score, label = 'Mean', color = color_cycle(2))
    plt.plot(gen, min_score, label = 'Minimum', color = color_cycle(0))
    hline(plt.gca(), max_score[-1], linestyle = '--', color = 'black', linewidth = 1)
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.legend(loc = 'lower right', fontsize = 13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
        plt.savefig('../article/figures/search/ising_max_history.pdf', bbox_inches='tight')
        
    

if __name__ == '__main__':
    # plot_ref_search(filename = '../ML/ref_search/pop_1_7_1_ref_search.npy', save = False)
    # plot_ref_search(filename = '../ML/ref_search/pop_5_3_1_ref_search.npy', save = False)
    # plot_ref_search(filename = '../ML/ref_search/hon_3_3_5_3_ref_search.npy', save = False)
    # plot_ref_search(filename = '../ML/ref_search/hon_2_3_3_3_ref_search.npy', save = False)
    # 
    # plot_ref_search(filename = '../ML/ref_search/hon_2_1_1_1_ref_search.npy')
    # plot_ref_search(filename = '../ML/ref_search/hon_2_1_5_3_ref_search.npy')
    # pass
    # plot_RW_top5(save = False)
    
    
    plot_search_history(filename = '../ML/best_ising_max_history.txt', save = False)
    
    plt.show()