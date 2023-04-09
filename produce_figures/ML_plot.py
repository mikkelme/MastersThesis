import sys
sys.path.append('../') # parent folder: MastersThesis
import matplotlib.pyplot as plt
import numpy as np

from ML.hypertuning import *
from ML.ML_perf import *
from analysis.analysis_utils import*
from matplotlib.gridspec import GridSpec

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
    
    ML_setting = {'use_gpu': False}


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
        
        ax.plot(lr[start_cut:], loss[start_cut:], color = color, label = f'{model.name:>5s} ({num_params:1.2e})') 
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
    
    
def LR_range_momentum(save = False):
    root = '../Data/ML_data/'
    data_root = [root+'baseline', root+'popup', root+'honeycomb', root+'RW']
    # data_root = [root+'honeycomb']

    ML_setting = {'use_gpu': False}
    
    
    start_lr = 1e-7
    end_lr = 10.0
    momentum = [0.85, 0.88, 0.91, 0.93, 0.95, 0.97, 0.99]
    lr_sug = []
    lr_max = []
    # filename = 'lr_momentum_test.txt'
    ML_setting = get_ML_setting()
    model, criterion = best_model(mode = 0, batchnorm = True)[0]
    state = model.state_dict() # Get model state for resetting
    
    
    fig1 = plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()
    ymin = 1e3; ymax = -1e3
    for i, mom in enumerate(momentum):
        print(f'{i+1}/{len(momentum)} | momentum = {mom}')
        optimizer = optim.Adam(model.parameters(), lr = start_lr, betas=(mom, 0.999))
        model.load_state_dict(state) # Reset model before training again
        foLR = Find_optimal_LR(model, optimizer, criterion, data_root, ML_setting)
        
        foLR.find_optimal(end_lr)
        
        # Plot
        results = foLR.lr_finder.get_results()
        lr = np.array(results['lr'])
        print('len(lr) = ', len(lr))
        
        loss = np.array(results['loss'])
        minidx = np.argmin(loss)
        sug_idx = np.argmin(np.abs(lr-foLR.lr_finder.lr_suggestion())) 
        # div_idx = sug_idx + np.argmax(loss[sug_idx:][1:] - loss[sug_idx:][:-1] > loss[sug_idx:][:-1]*0.2)
        div_idx = minidx
        
        # diff = loss[sug_idx:][1:] - loss[sug_idx:][:-1]
        # test = np.argwhere(diff > loss[sug_idx:][:-1]*1.5)
        
        start_cut = np.argmin(np.abs(lr - 1e-5))
        
        ax.plot(lr[start_cut:], loss[start_cut:], color = color_cycle(i), label = fr'$\beta_1 = ${mom:0.2f}') 
        ax.plot(lr[sug_idx], loss[sug_idx], color = color_cycle(i), marker = 'o')
        ax.plot(lr[div_idx], loss[div_idx], color = color_cycle(i), marker = 'o')
        
        ymin = np.min((loss[minidx], ymin))
        ymax = np.max((np.max(loss[start_cut:minidx]), ymax))
        ax.set_xlabel('Learning rate', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)


        # Store values
        lr_sug.append(lr[sug_idx])
        lr_max.append(lr[div_idx])
        
        
    print('momentum', momentum)
    print('lr_sug', lr_sug)
    print('lr_max', lr_max)
        
    diff = ymax - ymin
    sp = 0.025*diff
    ax.set_ylim([ymin - sp, ymax + sp]) # Adjust ylim
    ax.set_xscale('log')
    legend = plt.legend(fontsize = 13)
    
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)


    fig2 = plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(momentum[:len(lr_sug)], lr_sug, '-o', label = 'Constant suggestion')
    plt.plot(momentum[:len(lr_max)], lr_max, '-o', label = 'Max suggestion')
    plt.xlabel(r'Momentum ($\beta_1$)', fontsize=14)
    plt.ylabel('Learning rate', fontsize=14)
    plt.yscale('log')
    plt.legend(fontsize = 13)

    fig1.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    fig2.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    if save:
        fig1.savefig("../article/figures/ML/LR_momentum_test_a.pdf", bbox_inches="tight")
        fig2.savefig("../article/figures/ML/LR_momentum_test_b.pdf", bbox_inches="tight")
    

    
    
    
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
    
    
    fig, axes = plt.subplots(2, 2,  figsize = figsize)#, gridspec_kw ={'width_ratios': width_ratios})


    # R2 Val
    mat = 100*P['R2_0']
    vmin, vmax = get_vmin_max(mat, p = 0.98)
    # vmax = 100
    cmap = sns.cm.rocket
    ax = axes[0,0]
    annot = annotation_array(mat, fmt = '.02f', mask_swap = [mat < 0.01, '< 0.01'])
    sns.heatmap(mat, xticklabels = D_axis, yticklabels = S_axis, cbar_kws={'label': r'$R_2$ $\langle F_{\parallel} \rangle$ [$10^2$] '}, annot=annot, fmt='', vmin=vmin, vmax=vmax, cmap = cmap, ax=ax)


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
        


def A_search_compare_perf(path, save = False):
    # --- Compare set --- #
    honeycomb = '../Data/CONFIGS/honeycomb/'
    popup = '../Data/CONFIGS/popup/'
    
    honeycomb_folder = [honeycomb + 'hon_28',
                        honeycomb + 'hon_29',
                        honeycomb + 'hon_21',
                        honeycomb + 'hon_42',
                        honeycomb + 'hon_6',
                        honeycomb + 'hon_8',
                        honeycomb + 'hon_12',
                        honeycomb + 'hon_20',
                        honeycomb + 'hon_4',
                        honeycomb + 'hon_17']

    popup_folder =     [popup + 'pop_27',
                        popup + 'pop_25',
                        popup + 'pop_35',
                        popup + 'pop_48',
                        popup + 'pop_43',
                        popup + 'pop_46',
                        popup + 'pop_50',
                        popup + 'pop_28',
                        popup + 'pop_1',
                        popup + 'pop_58']
    
    compare_folder = honeycomb_folder + popup_folder


    # --- Get R2 Ff scores --- #
    # Get architecture data
    D_axis, S_axis, P = get_staircase_perf(path)
    
    # Evaluation on selected configurations
    paths = P['path']
    R2 = np.zeros((len(compare_folder), paths.shape[0], paths.shape[1]))
    for f, folder in enumerate(compare_folder):
        print(f'{f+1}/{len(compare_folder)}')
        
        # Get compare data
        data = read_multi_folder(folder)
        config_path = find_single_file(folder, '.npy')
        stretch = data['stretch_pct']
        F_N = data['F_N']    
        Ff = data['Ff'][:, :, 0, 1]
        
        for i in range(paths.shape[0]):
            for j in range(paths.shape[1]):
                if paths[i,j] == 'nan': continue
                model_weights = os.path.join(paths[i,j], 'model_dict_state')
                model_info = os.path.join(paths[i,j], 'best_scores.txt')
                EV = Evaluater(model_weights, model_info)
                EV.load_config(config_path)
                
        
                # Predict for different F_N
                Ff_pred = np.zeros((len(stretch), len(F_N)))
                no_rupture = np.zeros((len(stretch), len(F_N))) == -1
                for k in range(len(F_N)):
                    no_rupture[:, k] = ~np.isnan(Ff[:, k]) 
                    _, _, output = EV.predict(stretch, F_N[k])
                    Ff_pred[no_rupture[:, k], k] = output[no_rupture[:, k], 0]
                
                Ff_target = Ff[no_rupture].flatten()
                Ff_pred = Ff_pred[no_rupture].flatten()
                
                Ff_target_mean = np.mean(Ff_target)
                SS_res = np.sum((Ff_pred - Ff_target)**2)
                SS_tot = np.sum((Ff_target - Ff_target_mean)**2)
                R2[f, i, j] = 1 - SS_res/SS_tot
    
    # --- Plotting --- #
    figsize = (10, 4)
    fig, axes = plt.subplots(1, 2,  figsize = figsize)
    vmin, vmax = 60, 100
    
    # Tetrahedron
    ax = axes[0]
    ax.set_title('Tetrahedron')
    mat = 100*np.mean(R2[10:], axis = 0)
    # vmin, vmax = get_vmin_max(mat, p = 0.98)
    cmap = sns.cm.rocket
    annot = annotation_array(mat, fmt = '.02f', mask_swap = [mat < 0.01, '< 0.01'])
    sns.heatmap(mat, xticklabels = D_axis, yticklabels = S_axis, cbar_kws={'label': r'$R_2$ $\langle F_{\parallel} \rangle$ [$10^2$] '}, annot=annot, fmt='', vmin=vmin, vmax=vmax, cmap = cmap, ax=ax)
    
    # Honeycomb
    ax = axes[1]
    ax.set_title('Honeycomb')
    mat = 100*np.mean(R2[:10], axis = 0)
    # vmin, vmax = get_vmin_max(mat, p = 0.98)
    cmap = sns.cm.rocket
    annot = annotation_array(mat, fmt = '.02f', mask_swap = [mat < 0.01, '< 0.01'])
    sns.heatmap(mat, xticklabels = D_axis, yticklabels = S_axis, cbar_kws={'label': r'$R_2$ $\langle F_{\parallel} \rangle$ [$10^2$] '}, annot=annot, fmt='', vmin=vmin, vmax=vmax, cmap = cmap, ax=ax)
    
    
    # Suplot settings
    fig.supxlabel('Depth', fontsize = 14)
    fig.supylabel('Start num. channels', fontsize = 14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    if save:
        fig.savefig("../article/figures/ML/A_search_compare_perf.pdf", bbox_inches="tight")
        

    
    


    
def mom_weight_search_perf(path, save):
    M_axis, W_axis, P = get_mom_weight_perf(path)


    # --- Plotting --- #
    figsize = (10, 4)
    
    fig, axes = plt.subplots(1, 2,  figsize = figsize)


    # R2 Val
    mat = 100*P['R2_0']
    vmin, vmax = get_vmin_max(mat, p = 0.98)
    # vmax = 100
    cmap = sns.cm.rocket
    ax = axes[0]
    annot = annotation_array(mat, fmt = '.02f', mask_swap = [mat < 0.01, '< 0.01'])
    sns.heatmap(mat, xticklabels = M_axis, yticklabels = W_axis, cbar_kws={'label': r'$R_2$ $\langle F_{\parallel} \rangle$ [$10^2$] '}, annot=annot, fmt='', vmin=vmin, vmax=vmax, cmap = cmap, ax=ax)


    # Val loss
    mat = 100*P['val_tot_loss']
    vmin, vmax = get_vmin_max(mat, p = 0.70, increasing = False)
    cmap = sns.cm.rocket_r
    ax = axes[1]
    sns.heatmap(mat, xticklabels = M_axis, yticklabels = W_axis, cbar_kws={'label': r'Validation loss [$10^2$]'}, annot=True, fmt='.4g', vmin=vmin, vmax=vmax, cmap = cmap, ax=ax)
    
   
    fig.supxlabel('Momentum', fontsize = 14)
    fig.supylabel('Weight decay', fontsize = 14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    if save:
        fig.savefig("../article/figures/ML/mom_weight_search_perf.pdf", bbox_inches="tight")
        
 



def mom_weight_search_compare_perf(path, save = False):
    # --- Compare set --- #
    honeycomb = '../Data/CONFIGS/honeycomb/'
    popup = '../Data/CONFIGS/popup/'
    
    honeycomb_folder = [honeycomb + 'hon_28',
                        honeycomb + 'hon_29',
                        honeycomb + 'hon_21',
                        honeycomb + 'hon_42',
                        honeycomb + 'hon_6',
                        honeycomb + 'hon_8',
                        honeycomb + 'hon_12',
                        honeycomb + 'hon_20',
                        honeycomb + 'hon_4',
                        honeycomb + 'hon_17']

    popup_folder =     [popup + 'pop_27',
                        popup + 'pop_25',
                        popup + 'pop_35',
                        popup + 'pop_48',
                        popup + 'pop_43',
                        popup + 'pop_46',
                        popup + 'pop_50',
                        popup + 'pop_28',
                        popup + 'pop_1',
                        popup + 'pop_58']
    
    compare_folder = honeycomb_folder + popup_folder


    # --- Get R2 Ff scores --- #
    # Get architecture data
    M_axis, W_axis, P = get_mom_weight_perf(path)
    
    # Evaluation on selected configurations
    paths = P['path']
    R2 = np.zeros((len(compare_folder), paths.shape[0], paths.shape[1]))
    for f, folder in enumerate(compare_folder):
        print(f'{f+1}/{len(compare_folder)}')
        
        # Get compare data
        data = read_multi_folder(folder)
        config_path = find_single_file(folder, '.npy')
        stretch = data['stretch_pct']
        F_N = data['F_N']    
        Ff = data['Ff'][:, :, 0, 1]
        
        for i in range(paths.shape[0]):
            for j in range(paths.shape[1]):
                if paths[i,j] == 'nan': continue
                model_weights = os.path.join(paths[i,j], 'model_dict_state')
                model_info = os.path.join(paths[i,j], 'best_scores.txt')
                EV = Evaluater(model_weights, model_info)
                EV.load_config(config_path)
                
        
                # Predict for different F_N
                Ff_pred = np.zeros((len(stretch), len(F_N)))
                no_rupture = np.zeros((len(stretch), len(F_N))) == -1
                for k in range(len(F_N)):
                    no_rupture[:, k] = ~np.isnan(Ff[:, k]) 
                    _, _, output = EV.predict(stretch, F_N[k])
                    Ff_pred[no_rupture[:, k], k] = output[no_rupture[:, k], 0]
                
                Ff_target = Ff[no_rupture].flatten()
                Ff_pred = Ff_pred[no_rupture].flatten()
                
                Ff_target_mean = np.mean(Ff_target)
                SS_res = np.sum((Ff_pred - Ff_target)**2)
                SS_tot = np.sum((Ff_target - Ff_target_mean)**2)
                R2[f, i, j] = 1 - SS_res/SS_tot
    
    # --- Plotting --- #
    figsize = (10, 4)
    fig, axes = plt.subplots(1, 2,  figsize = figsize)
    vmin, vmax = 60, 100
    
    # Tetrahedron
    ax = axes[0]
    ax.set_title('Tetrahedron')
    mat = 100*np.mean(R2[10:], axis = 0)
    # vmin, vmax = get_vmin_max(mat, p = 0.98)
    cmap = sns.cm.rocket
    annot = annotation_array(mat, fmt = '.02f', mask_swap = [mat < 0.01, '< 0.01'])
    sns.heatmap(mat, xticklabels = M_axis, yticklabels = W_axis, cbar_kws={'label': r'$R_2$ $\langle F_{\parallel} \rangle$ [$10^2$] '}, annot=annot, fmt='', vmin=vmin, vmax=vmax, cmap = cmap, ax=ax)
    
    # Honeycomb
    ax = axes[1]
    ax.set_title('Honeycomb')
    mat = 100*np.mean(R2[:10], axis = 0)
    # vmin, vmax = get_vmin_max(mat, p = 0.98)
    cmap = sns.cm.rocket
    annot = annotation_array(mat, fmt = '.02f', mask_swap = [mat < 0.01, '< 0.01'])
    sns.heatmap(mat, xticklabels = M_axis, yticklabels = W_axis, cbar_kws={'label': r'$R_2$ $\langle F_{\parallel} \rangle$ [$10^2$] '}, annot=annot, fmt='', vmin=vmin, vmax=vmax, cmap = cmap, ax=ax)
    
    
    # Suplot settings
    fig.supxlabel('Momentum', fontsize = 14)
    fig.supylabel('Weight decay', fontsize = 14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)

    
    if save:
        fig.savefig("../article/figures/ML/mom_weight_search_compare_perf.pdf", bbox_inches="tight")
        

    

def final_model_evaluation(model_path, save = False):
    """ Evaluate the final model on pilot study tetrehedron and honeycomb """

    model_weights = os.path.join(model_path, 'model_dict_state')
    model_info = os.path.join(model_path, 'best_scores.txt')
    EV = Evaluater(model_weights, model_info)


    # Pilot study patterns
    folders = ['../Data/Baseline_fixmove/popup/multi_stretch',
               '../Data/Baseline_fixmove/honeycomb/multi_stretch']
    patterns = ['Tetrahedron', 'Honeycomb']
    
    colorbar_scale = [(0.1, 10), 'log']
    num_points = 100 # ML
    
    cmap = matplotlib.cm.viridis
    line_and_marker = {'linestyle': '', 
                       'marker': 'o',
                       'linewidth': 1.5,
                       'markersize': 2.5}
    


    # fig, axes = plt.subplots(3, 2, num = unique_fignum(), figsize = (10,8))
    
    nrow = 3; ncol = 2
    fig = plt.figure(figsize = (10, 8), constrained_layout=True)
    gs = GridSpec(nrow, ncol + 1, figure=fig, width_ratios=[1,1,0.05])
    axes = []
    for i in range(nrow):
        axes.append([fig.add_subplot(gs[i, j]) for j in range(ncol)])

    
    

    for f, folder in enumerate(folders):
        # Data
        data = read_multi_folder(folder, mean_pct = 0.5, std_pct = 0.35)
        stretch = data['stretch_pct']
        F_N = data['F_N']
        Ff_mean = data['Ff'][:, :, 0, 1]
        Ff_max = data['Ff'][:, :, 0, 0]
        contact = data['contact_mean'][:, :, 0]
        rup = data['rup']
        rupture_stretch = (data['rupture_stretch'], data['practical_rupture_stretch'])
        
    
        # Prepare ML 
        stretch_space = np.linspace(np.min(stretch), np.max(stretch), num_points)
        config_path = find_single_file(folder, '.npy')
        EV.load_config(config_path)
    
    
        # Go through different outputs    
        Z = [Ff_mean, Ff_max, contact]
        ylabel = [r'$\langle F_\parallel \rangle$ [nN]', 
                  r'$\max \ F_\parallel$ [nN]',
                  r'Rel. $\langle$ Bond $\rangle$']
        
        
        axes[0][f].set_title(f'{patterns[f]}')
        axes[-1][f].set_xlabel('Stretch', fontsize = 14)     
        for o in range(3): # Ff_mean, Ff_max, contact
            ax = axes[o][f]
            if f == 0:
                ax.set_ylabel(ylabel[o], fontsize = 14)
            
                
            # Go through load 
            z = Z[o]
            no_rupture_data = ~np.isnan(z[:, 0]) 
            for k in range(len(F_N)):    
                color = get_color_value(F_N[k], colorbar_scale[0][0], colorbar_scale[0][1], scale = colorbar_scale[-1], cmap = cmap)
                
                # --- Simulation --- #
                ax.plot(stretch, z[:,k], **line_and_marker, color = color)
                
                # Prepare R2
                target = z[no_rupture_data, k]
                target_mean = np.mean(target)
                
                
                # --- ML --- #
                _, _, output = EV.predict(stretch, F_N[k]) # ['Ff_mean', 'Ff_max', 'contact', 'porosity', 'rupture_stretch', 'is_ruptured']
                pred = output[no_rupture_data, o]
                SS_res = np.sum((pred - target)**2)
                SS_tot = np.sum((target - target_mean)**2)
                R2 = 1 - SS_res/SS_tot
            
            
                # Produce more smooth stretch curve fore plotting
                _, _, output = EV.predict(stretch_space, F_N[k])
                no_rupture = output[:,-1] < 0.5
                color = get_color_value(F_N[k], colorbar_scale[0][0], colorbar_scale[0][1], scale = colorbar_scale[1], cmap = cmap)
                ax.plot(stretch_space[no_rupture], output[no_rupture, o], color = color, label = rf'$R_2$ = {R2:0.4f}', zorder = -1)
                ax.legend(fontsize = 13)
   
    # Colorbar 
    norm = matplotlib.colors.LogNorm(colorbar_scale[0][0], colorbar_scale[0][1])
    axes.append(fig.add_subplot(gs[:, ncol]))
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax = axes[-1])
    cb.set_label(label = r'$F_N$ [nN]', fontsize=14)
    
    
    # Adjust 
    # fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    if save:
        fig.savefig("../article/figures/ML/final_model_evaluation.pdf", bbox_inches="tight")
        

        


if __name__ == '__main__':
    # LR_range_specific(A_staircase_subset(mode = 0, batchnorm = True), save = False)
    # LR_range_full(filename = '../ML/staircase_lr/lr.txt', save = False)
    # LR_range_momentum(save = True)
    
    # A_search_perf(path = '../ML/staircase_4', save = True)
    # A_search_compare_perf(path = '../ML/staircase_4', save = True)
    
    
    mom_weight_search_perf(path = '../ML/mom_weight_search', save =  True)
    mom_weight_search_compare_perf(path = '../ML/mom_weight_search', save =  True)
    
    
    # final_model_evaluation(model_path = '../ML/staircase_4/S32D12', save = True)
    plt.show()



# Honeycomb
# Extrema: Max drop
# (2, 3, 3, 3)         hon_28/hon3333.npy   0.541 1.01 1.279 
# (2, 1, 3, 1)         hon_29/hon3131.npy   0.8611 1.616 1.105 
# (2, 3, 3, 5)         hon_21/hon3335.npy   0.5309 0.9056 0.8947 
# (2, 1, 5, 3)         hon_42/hon3153.npy   1.476 1.922 0.8638 
# (2, 5, 1, 1)         hon_6/hon3511.npy    0.3195 0.3741 0.8468 
# (2, 3, 1, 5)         hon_8/hon3315.npy    0.5767 0.7644 0.8114 
# (2, 1, 1, 1)         hon_12/hon3111.npy   1.065 1.712 0.8048 
# (2, 4, 5, 3)         hon_20/hon3453.npy   0.3944 0.7312 0.6835 
# (2, 2, 1, 3)         hon_4/hon3213.npy    0.7186 1.129 0.6548 
# (2, 5, 3, 1)         hon_17/hon3531.npy   0.3251 0.4905 0.6406 

# Popup
# Extrema: Max drop
# (5, 3, 1)            pop_27/pop1_5_3.npy  0.1391 0.1999 0.8841 
# (3, 5, 1)            pop_25/pop1_3_5.npy  0.0786 0.1597 0.4091 
# (7, 5, 1)            pop_35/pop1_7_5.npy  0.0807 0.1142 0.3775 
# (9, 7, 1)            pop_48/pop1_9_7.npy  0.0724 0.1342 0.2238 
# (1, 1, 1)            pop_43/pop1_1_11.npy 0.0004 0.0368 0.1347 
# (3, 1, 4)            pop_46/pop4_3_1.npy  0.1211 0.1598 0.1007 
# (3, 9, 1)            pop_50/pop1_3_9.npy  0.0885 0.1086 0.0993 
# (7, 1, 1)            pop_28/pop1_7_1.npy  0.1639 0.1846 0.0913 
# (5, 3, 2)            pop_1/pop2_5_3.npy   0.0721 0.1087 0.0874 
# (7, 3, 1)            pop_58/pop1_7_13.npy 0.0574 0.0942 0.0873 