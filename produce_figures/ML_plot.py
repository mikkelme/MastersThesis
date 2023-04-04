import sys
sys.path.append('../') # parent folder: MastersThesis
import matplotlib.pyplot as plt
import numpy as np

from ML.hypertuning import *
from ML.ML_perf import *
from analysis.analysis_utils import*



class A_staircase_subset(Architectures):    

    def initialize(self):
        # Data outputs
        alpha = [[1/2, 1/10, 1/10], [1/10], [1/10, 1/10]]
        criterion_out_features = [['R', 'R', 'R'], ['R'], ['R', 'C']]
        keys = ['Ff_mean', 'Ff_max', 'contact', 'porosity', 'rupture_stretch', 'is_ruptured']
        model_out_features = [item for sublist in criterion_out_features for item in sublist]   
        criterion = Loss(alpha = alpha, out_features = criterion_out_features)
    
        # Fill with architectures
        start = [2, 4, 8, 16, 32, 64, 128, 256] # Number of channels for first layer
        depth = [4, 6, 8, 10, 12, 14] # Number of CNN and FC layers (excluding final FC to output)
        
        start_depth = [(2,8), (4,8), (8,8), (16,8)]
        for (s, d) in start_depth:
            name = f'S{s}D{d}'
            conv_layers = [(1, s*2**x) for x in range(d//2)]
            FC_layers = [(1, s*2**x) for x in reversed(range(d//2))] 
            model = VGGNet( name = name,
                            mode = self.mode, 
                            input_num = 2, 
                            conv_layers = conv_layers, 
                            FC_layers = FC_layers,
                            out_features = model_out_features,
                            keys = keys,
                            batchnorm = self.batchnorm)
            
            # Add to list of architectures
            self.A.append((model, criterion)) 
            

def LR_range_specific(A_instance, save = False):
    start_lr = 1e-7
    end_lr = 10.0
    
    root = '../Data/ML_data/'
    data_root = [root+'baseline', root+'popup', root+'honeycomb', root+'RW']
    # data_root = [root+'honeycomb']
    
    ML_setting = {
        'use_gpu': False,
        'lr': 0.0001,
        'batchsize_train': 32,
        'batchsize_val': 64,
        'max_epochs': 1000,
        'max_file_num': None,
        'scheduler_stepsize': None, 
        'scheduler_factor': None
    }


    fig = plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()
    ymin = 1e3; ymax = -1e3
    for i, (model, criterion) in enumerate(A_instance):
        num_params = model.get_num_params()
        print(f'{i} | {model.name} (#params = {num_params:1.2e})')

        # Perform LR range test 
        optimizer = optim.Adam(model.parameters(), lr = start_lr)
        foLR = Find_optimal_LR(model, optimizer, criterion, data_root, ML_setting)
        foLR.find_optimal(end_lr)
        
        # Plot
        results = foLR.lr_finder.get_results()
        lr = np.array(results['lr'])
        loss = np.array(results['loss'])
        sug_idx = np.argmin(np.abs(lr-foLR.lr_finder.lr_suggestion()))
        
        start_cut = np.argmin(np.abs(lr - 1e-5))
        minidx = np.argmin(loss)
        
        color = get_color_value(int(model.name.split('D')[0].strip('S')), 2, 256, scale = 'log', cmap='viridis')
        
        ax.plot(lr[start_cut:], loss[start_cut:], color = color, label = f'{model.name:>5s} ({num_params:1.2e})') # Plot start
        ax.plot(lr[sug_idx], loss[sug_idx], color = color, marker = 'o')
        
        ymin = np.min((loss[minidx], ymin))
        ymax = np.max((np.max(loss[start_cut:minidx]), ymax))
        ax.set_xlabel('Learning rate', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        
        
    diff = ymax - ymin
    sp = 0.025*diff
    ax.set_ylim([ymin - sp, ymax + sp]) # Adjust ylim
    ax.set_xscale('log')
    plt.legend(fontsize = 13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    if save:
        plt.savefig("../article/figures/ML/LR_range_specific.pdf", bbox_inches="tight")
    
    
def LR_range_full(filename, save = False):
    num_params = []; lr = []; S = []; D = []
    infile = open(filename, 'r')
    for line in infile:
        if line[0] == '#': continue        
        name = line.split('|')[-1].split('(')[0].strip(' ')
        S.append(int(name.split('D')[0].strip('S')))
        D.append(int(name.split('D')[1].strip('D')))
        num_params.append(float(line.split('params = ')[1].split(')')[0]))
        lr.append(float(line.split('lr = ')[-1]))
    
    num_params = np.array(num_params)
    lr = np.array(lr)
    S = np.array(S)
    D = np.array(D)
    
    D_set =  np.sort(list(set(D)))
    S_set =  np.sort(list(set(S)))


    size_ratio_label = 12
    cmap = 'viridis'
    # cmap = 'cividis'

    sizes = size_ratio_label*D
    # sizes = [60]*len(D)
    colors = get_color_value(S, np.min(S), np.max(S), scale = 'log', cmap=cmap)
    markers = np.array([{4:(3, 0, 0), 6:(4, 0, 0), 8:(5, 0, 0), 10:(4, 1, 0), 12:(5, 1, 0), 14:(6, 1, 0)}[d] for d in D])
    
    fig = plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()
    
    
    
    # Make connective lines
    for d in D_set:
        idx = np.argwhere(d == D)
        sort = idx[np.argsort(num_params[idx].ravel())]
        ax.plot(num_params[sort], lr[sort], linewidth = 0.5, linestyle = '--', color = 'black', alpha = 0.3, zorder = -1)
        
    for i in range(len(D)):
        ax.scatter(num_params[i], lr[i], marker = markers[i], s = sizes[i], color = colors[i])
        
        
    # Add labels
    for i, d in enumerate(D_set):
        size = size_ratio_label*d
        ax.scatter([], [], color = 'grey', marker = markers[i], s = size, label = f'=  {d:2d}')
    h, l = ax.get_legend_handles_labels()
    legend = ax.legend(h, l, loc='upper right',  handletextpad=0.00, fontsize = 13)
    legend.set_title("Depth", {'size': 13})
    
    norm = matplotlib.colors.Normalize(0.5, len(S_set)+0.5)
    bounds = np.linspace(0.5, len(S_set)+0.5, len(S_set)+1)
    cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), spacing='proportional', boundaries=bounds, ticks=np.arange(1, len(S_set)+1), drawedges = False, ax = ax)    
    cb.ax.set_yticklabels(S_set) 
    cb.outline.set_edgecolor('black')
    cb.solids.set_edgecolor('black')
    cb.set_label(label = 'Start num. channels', fontsize=14)
    

    # Adjust axis      
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Num. parameters', fontsize=14)
    plt.ylabel('Learning rate', fontsize=14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    if save:
        plt.savefig("../article/figures/ML/LR_range_full.pdf", bbox_inches="tight")
    
        
def annotation_array(mat, fmt, mask_swap = None):
    annot = np.empty(np.shape(mat), dtype="<U10")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            annot[i, j] = f'{mat[i,j]:{fmt}}'
    
    if mask_swap:
        annot[mask_swap[0]] = mask_swap[1]
    
    return annot

def A_search_perf(path, save):
    D_axis, S_axis, P = get_staircase_perf(path)

    # --- Plotting --- #
    figsize = (10, 6)
    figsize = figsize
    
    
    fig, axes = plt.subplots(2, 2,  figsize = figsize)#, gridspec_kw ={'width_ratios': width_ratios})


    # R2 Val
    mat = 100*P['R2_0']
    vmin, vmax = get_vmin_max(mat, p = 0.98)
    cmap = sns.cm.rocket
    ax = axes[0,0]
    # annot = annotation_array(mat, fmt = '.02f', mask_swap = [mat < 1e-3, r'$\sim$ 0'])
    annot = annotation_array(mat, fmt = '.02f', mask_swap = [mat < 0.01, '< 0.01'])
    sns.heatmap(mat, xticklabels = D_axis, yticklabels = S_axis, cbar_kws={'label': r'$R_2$ $\langle F_{\parallel} \rangle$ [$10^2$] '}, annot=annot, fmt='', vmin=vmin, vmax=vmax, cmap = cmap, ax=ax)
    # sns.heatmap(mat, xticklabels = D_axis, yticklabels = S_axis, cbar_kws={'label': r'$R_2$ $\langle F_{\parallel} \rangle$ '}, annot=annot, fmt='.4g', vmin=vmin, vmax=vmax, cmap = cmap, ax=ax)


    # Val loss
    mat = 100*P['val_tot_loss']
    vmin, vmax = get_vmin_max(mat, p = 0.70, increasing = False)
    cmap = sns.cm.rocket_r
    ax = axes[0,1]
    sns.heatmap(mat, xticklabels = D_axis, yticklabels = S_axis, cbar_kws={'label': r'Validation loss [$10^2$]'}, annot=True, fmt='.4g', vmin=vmin, vmax=vmax, cmap = cmap, ax=ax)
    
    # Epoch
    mat = P['epoch']
    vmin, vmax = 0, 1000
    cmap = sns.cm.rocket
    ax = axes[1,0]
    sns.heatmap(mat, xticklabels = D_axis, yticklabels = S_axis, cbar_kws={'label': 'Best epoch'}, annot=True, fmt='.3g', vmin=vmin, vmax=vmax, cmap = cmap, ax=ax)
    
    # Num. Params
    mat = P['num_params']
    vmin, vmax = None, None
    cmap = sns.cm.rocket
    ax = axes[1,1]
    sns.heatmap(mat, xticklabels = D_axis, yticklabels = S_axis, cbar_kws={'label': 'Num. parameters'}, annot=True, fmt='.2g', vmin=vmin, vmax=vmax, cmap=cmap, ax=ax, norm=LogNorm())
   
    fig.supxlabel('Depth', fontsize = 14)
    fig.supylabel('Start num. channels', fontsize = 14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    
    if save:
        
        fig.savefig("../article/figures/ML/A_search_perf.pdf", bbox_inches="tight")
        
   
    return


    # R2 Val
    R2_fig, ax = plt.subplots(num = unique_fignum(), figsize=figsize)
    mat = P['R2_0']
    vmin, vmax = get_vmin_max(mat, p = 0.98)
    cmap = sns.cm.rocket
    sns.heatmap(mat, xticklabels = D_axis, yticklabels = S_axis, cbar_kws={'label': r'$R_2$ $\langle F_{\parallel} \rangle$ '}, annot=True, fmt='.3g', vmin=vmin, vmax=vmax, cmap = cmap, ax=ax)
    ax.set_xlabel('Depth', fontsize=14)
    ax.set_ylabel('Start num. channels', fontsize=14)
    R2_fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    # Val loss
    val_loss_fig, ax = plt.subplots(num = unique_fignum(),  figsize=figsize)
    mat = P['val_tot_loss']
    vmin, vmax = get_vmin_max(mat, p = 0.70, increasing = False)
    cmap = sns.cm.rocket_r
    sns.heatmap(mat, xticklabels = D_axis, yticklabels = S_axis, cbar_kws={'label': 'Validation loss'}, annot=True, fmt='.3g', vmin=vmin, vmax=vmax, cmap = cmap, ax=ax)
    ax.set_xlabel('Depth', fontsize=14)
    ax.set_ylabel('Start num. channels', fontsize=14)
    val_loss_fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    # Epoch
    epoch_fig, ax = plt.subplots(num = unique_fignum(),  figsize=figsize)
    mat = P['epoch']
    vmin, vmax = 0, 1000
    cmap = sns.cm.rocket
    sns.heatmap(mat, xticklabels = D_axis, yticklabels = S_axis, cbar_kws={'label': 'Best epoch'}, annot=True, fmt='.3g', vmin=vmin, vmax=vmax, cmap = cmap, ax=ax)
    ax.set_xlabel('Depth', fontsize=14)
    ax.set_ylabel('Start num. channels', fontsize=14)
    epoch_fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    # Num. Params
    num_param_fig, ax = plt.subplots(num = unique_fignum(),  figsize=figsize)
    mat = P['num_params']
    vmin, vmax = None, None
    cmap = sns.cm.rocket
    sns.heatmap(mat, xticklabels = D_axis, yticklabels = S_axis, cbar_kws={'label': 'Num. parameters'}, annot=True, fmt='.2g', vmin=vmin, vmax=vmax, cmap=cmap, ax=ax, norm=LogNorm())
    ax.set_xlabel('Depth', fontsize=14)
    ax.set_ylabel('Start num. channels', fontsize=14)
    num_param_fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)

        
   
    if save:
        R2_fig.savefig("../article/figures/ML/A_search_R2.pdf", bbox_inches="tight")
        val_loss_fig.savefig("../article/figures/ML/A_search_val_loss.pdf", bbox_inches="tight")
        epoch_fig.savefig("../article/figures/ML/A_search_epoch.pdf", bbox_inches="tight")
        num_param_fig.savefig("../article/figures/ML/A_search_num_param.pdf", bbox_inches="tight")

        


    




if __name__ == '__main__':
    # LR_range_specific(A_staircase_subset(mode = 0, batchnorm = True), save = False)
    # LR_range_full(filename = '../ML/staircase_lr/lr.txt', save = False)
    A_search_perf(path = '../ML/staircase_4', save = True)
    
    plt.show()
