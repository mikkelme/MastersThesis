### Evaluate model performance


import sys
sys.path.append('../') # parent folder: MastersThesis

from ML.data_analysis import *
from ML.use_network import *
from plot_set import *
import ast
from matplotlib.colors import LogNorm

def get_vmin_max(mat, p = 0.5, increasing = True):
    if increasing:
        vmin = np.nanmax((p*np.nanmax(mat), np.nanmin(mat)))
        vmax = np.nanmax(mat)
    else:
        vmin = np.nanmin(mat)
        vmax = np.nanmin((np.nanmin(mat)/p, np.nanmax(mat)))
    return vmin, vmax


def get_staircase_perf(path):
    folders = os.listdir(path)
    
    S = [] # Start
    D = [] # Depth
    P = {}
    for folder in folders:
        if '.DS_Store' in folder or '.txt' in folder:
            continue 
        
        file = os.path.join(path, folder, 'best_scores.txt')
        info, data = read_best_scores(file)
        
        conv_layers = info['Model_settings']['conv_layers']
        FC_layers = info['Model_settings']['FC_layers']
        assert len(conv_layers) == len(FC_layers), f'len of CNN layers = {len(conv_layers)} does not match len of FC layers = {len(FC_layers)}.'
        
        d = 2*len(conv_layers)
        s = conv_layers[0][1]
        D.append(d)
        S.append(s)
        
        for key in data:
            try:
                P[key].append(data[key])
            except KeyError:
                P[key] = [data[key]]    
        try:
            P['num_params'].append(info['Model_info']['num_params'])
        except KeyError:
            P['num_params'] = [info['Model_info']['num_params']]
            
        try:
            P['path'].append(os.path.join(path, folder))
        except KeyError:
            P['path'] = [os.path.join(path, folder)]
                

    # --- Organize into matrix (D, S, P) --- #
    # Get unique axis
    D_axis =  np.sort(list(set(D)))
    S_axis =  np.sort(list(set(S)))
    shape = (len(D_axis), len(S_axis))
    
    # Get 1D -> 2D mapping
    D_mat, S_mat = np.meshgrid(D_axis, S_axis)
    map = np.full(np.shape(D_mat), -1)
    for i in range(D_mat.shape[0]):
        for j in range(D_mat.shape[1]):
            D_hit = D_mat[i, j] == D
            S_hit = S_mat[i, j] == S
            full_hit = np.logical_and(D_hit, S_hit)
            if np.sum(full_hit) == 1:
                map[i,j] = int(np.argmax(full_hit))
            elif np.sum(full_hit) > 1:
                exit('This should not happen')
                
    # Flip axis for increasing y-axis
    map = np.flip(map, axis = 0)
    S_axis = np.flip(S_axis) 
    
    # Perform mapping
    D = np.array(D + [np.nan])[map]
    S = np.array(S + [np.nan])[map]
    for key in P:  
        P[key] = np.array(P[key]+[np.nan])[map]
    
   
    return D_axis, S_axis, P
   
   
def get_mom_weight_perf(path):
    folders = os.listdir(path)
    
    M = [] # Momentum
    W = [] # Weight decay
    P = {}
    for folder in folders:
        if '.DS_Store' in folder or '.txt' in folder:
            continue 
        
        file = os.path.join(path, folder, 'best_scores.txt')
        info, data = read_best_scores(file)
        ML_settings = info['ML_settings']
        if ML_settings['cyclic_momentum'] == 'None':
            M.append(float(ML_settings['momentum']))
        else:
            M.append(float(ML_settings['cyclic_momentum'][1]))
            
        W.append(float(ML_settings['weight_decay']))
        
        
        for key in data:
            try:
                P[key].append(data[key])
            except KeyError:
                P[key] = [data[key]]    
        try:
            P['num_params'].append(info['Model_info']['num_params'])
        except KeyError:
            P['num_params'] = [info['Model_info']['num_params']]
            
        try:
            P['path'].append(os.path.join(path, folder))
        except KeyError:
            P['path'] = [os.path.join(path, folder)]
                


    # --- Organize into matrix (M, W, P) --- #
    # Get unique axis
    M_axis =  np.sort(list(set(M)))
    W_axis =  np.sort(list(set(W)))
    shape = (len(M_axis), len(W_axis))
    
    
    
    
    # Get 1D -> 2D mapping
    M_mat, W_mat = np.meshgrid(M_axis, W_axis)
        
    map = np.full(np.shape(M_mat), -1)
    for i in range(M_mat.shape[0]):
        for j in range(M_mat.shape[1]):
            M_hit = M_mat[i, j] == M
            W_hit = W_mat[i, j] == W
            full_hit = np.logical_and(M_hit, W_hit)
            if np.sum(full_hit) == 1:
                map[i,j] = int(np.argmax(full_hit))
            elif np.sum(full_hit) > 1:
                exit('This should not happen')
    
   
    
    # Flip axis for increasing y-axis
    map = np.flip(map, axis = 0)
    W_axis = np.flip(W_axis) 
    
    # Perform mapping
    M = np.array(M + [np.nan])[map]
    W = np.array(W + [np.nan])[map]
    for key in P:  
        P[key] = np.array(P[key]+[np.nan])[map]
    

    return M_axis, W_axis, P
   

def mom_weight_heatmap(path, compare_folder = [], save = False):
    M_axis, W_axis, P = get_mom_weight_perf(path)
    
    
    # Evaluation on selected configurations
    if len(compare_folder) > 0:
        paths = P['path']
        R2 = np.zeros((len(compare_folder), paths.shape[0], paths.shape[1]))
        for f, folder in enumerate(compare_folder):
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
        
        
            # R2 test
            fig, ax = plt.subplots(num = unique_fignum(), figsize=(10, 6))
            mat = R2[f]
            vmin, vmax = get_vmin_max(mat)
            sns.heatmap(mat, xticklabels = M_axis, yticklabels = W_axis, cbar_kws={'label': fr'$R_2$ test {f}'}, annot=True, fmt='.4g', vmin=vmin, vmax=vmax, ax=ax)
            ax.set_xlabel('Momentum', fontsize=14)
            ax.set_ylabel('Weight decay', fontsize=14)
            fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
          



    # --- Plotting --- #
    # R2 Val
    fig, ax = plt.subplots(num = unique_fignum(), figsize=(10, 6))
    mat = P['R2_0']
    vmin, vmax = get_vmin_max(mat)
    sns.heatmap(mat, xticklabels = M_axis, yticklabels = W_axis, cbar_kws={'label': r'$R_2$ $\langle F_{\parallel} \rangle$ '}, annot=True, fmt='.4g', vmin=vmin, vmax=vmax, ax=ax)
    ax.set_xlabel('Momentum', fontsize=14)
    ax.set_ylabel('Weight decay', fontsize=14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    
    # Loss
    fig, ax = plt.subplots(num = unique_fignum(),  figsize=(10, 6))
    mat = P['val_tot_loss']
    vmin = np.nanmin(mat)
    vmax = np.nanmin((0.03, np.nanmax(mat)))
    sns.heatmap(mat, xticklabels = M_axis, yticklabels = W_axis, cbar_kws={'label': 'Validation loss'}, annot=True, fmt='.4g', vmin=vmin, vmax=vmax, ax=ax, cmap = sns.cm.rocket_r)
    ax.set_xlabel('Momentum', fontsize=14)
    ax.set_ylabel('Weight decay', fontsize=14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    # Epoch
    fig, ax = plt.subplots(num = unique_fignum(),  figsize=(10, 6))
    mat = P['epoch']
    sns.heatmap(mat, xticklabels = M_axis, yticklabels = W_axis, cbar_kws={'label': 'Epoch'}, annot=True, fmt='.4g', vmin=None, vmax=None, ax=ax)
    ax.set_xlabel('Momentum', fontsize=14)
    ax.set_ylabel('Weight decay', fontsize=14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    



def staircase_heatmap(path, compare_folder = [], save = False):
    D_axis, S_axis, P = get_staircase_perf(path)

    # Evaluation on selected configurations
    if len(compare_folder) > 0:
        paths = P['path']
        R2 = np.zeros((len(compare_folder), paths.shape[0], paths.shape[1]))
        for f, folder in enumerate(compare_folder):
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
        
        
            # R2 test
            fig, ax = plt.subplots(num = unique_fignum(), figsize=(10, 6))
            mat = R2[f]
            vmin, vmax = get_vmin_max(mat)
            sns.heatmap(mat, xticklabels = D_axis, yticklabels = S_axis, cbar_kws={'label': fr'$R_2$ test {f}'}, annot=True, fmt='.4g', vmin=vmin, vmax=vmax, ax=ax)
            ax.set_xlabel('Depth', fontsize=14)
            ax.set_ylabel('Start num. channels', fontsize=14)
            fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
          



    # --- Plotting --- #
    # R2 Val
    fig, ax = plt.subplots(num = unique_fignum(), figsize=(10, 6))
    mat = P['R2_0']
    vmin, vmax = get_vmin_max(mat)
    sns.heatmap(mat, xticklabels = D_axis, yticklabels = S_axis, cbar_kws={'label': r'$R_2$ $\langle F_{\parallel} \rangle$ '}, annot=True, fmt='.4g', vmin=vmin, vmax=vmax, ax=ax)
    ax.set_xlabel('Depth', fontsize=14)
    ax.set_ylabel('Start num. channels', fontsize=14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    
    # Loss
    fig, ax = plt.subplots(num = unique_fignum(),  figsize=(10, 6))
    mat = P['val_tot_loss']
    vmin = np.nanmin(mat)
    vmax = np.nanmin((0.03, np.nanmax(mat)))
    sns.heatmap(mat, xticklabels = D_axis, yticklabels = S_axis, cbar_kws={'label': 'Validation loss'}, annot=True, fmt='.4g', vmin=vmin, vmax=vmax, ax=ax)
    ax.set_xlabel('Depth', fontsize=14)
    ax.set_ylabel('Start num. channels', fontsize=14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    # Epoch
    fig, ax = plt.subplots(num = unique_fignum(),  figsize=(10, 6))
    mat = P['epoch']
    sns.heatmap(mat, xticklabels = D_axis, yticklabels = S_axis, cbar_kws={'label': 'Epoch'}, annot=True, fmt='.4g', vmin=None, vmax=None, ax=ax)
    ax.set_xlabel('Depth', fontsize=14)
    ax.set_ylabel('Start num. channels', fontsize=14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    # Num. Params
    fig, ax = plt.subplots(num = unique_fignum(),  figsize=(10, 6))
    mat = P['num_params']
    sns.heatmap(mat, xticklabels = D_axis, yticklabels = S_axis, cbar_kws={'label': 'Num. params'}, annot=True, fmt='.4g', vmin=None, vmax=None, ax=ax, norm=LogNorm())
    ax.set_xlabel('Depth', fontsize=14)
    ax.set_ylabel('Start num. channels', fontsize=14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)

        
    

    if save:
        pass
        # plt.savefig('../article/figures/figure.pdf', bbox_inches='tight')


def staircase_complexity(path):
    D_axis, S_axis, P = get_staircase_perf(path)
    
    # Overfitting heatmnap
    fig, ax = plt.subplots(num = unique_fignum(),  figsize=(10, 6))
    diff = np.abs(P['val_tot_loss'] - P['train_tot_loss'])
    sns.heatmap(diff, xticklabels = D_axis, yticklabels = S_axis, cbar_kws={'label': 'Loss diff: Val - Train'}, annot=True, fmt='.4g', vmin=None, vmax=None, ax=ax)
    ax.set_xlabel('Depth', fontsize=14)
    ax.set_ylabel('Start num. channels', fontsize=14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)


    
    num_params = P['num_params']
    sort = np.argsort(num_params.flatten())
    for key in P:
        P[key] = (P[key].flatten())[sort]
    
    
    fig = plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.gca()
    
    plt.plot(P['num_params'], P['train_tot_loss'], label = 'train tot loss')
    plt.plot(P['num_params'], P['val_tot_loss'], label = 'val tot loss')
    ax.set_xscale('log')
    fig.legend()
    
    
    
if __name__ == '__main__':
    path = '../ML/mom_weight_search'
    
    compare_folder = ['../Data/Baseline_fixmove/honeycomb/multi_stretch', '../Data/Baseline_fixmove/popup/multi_stretch']
    staircase_heatmap(path, compare_folder, save = False)
    # staircase_complexity(path)
    
    # mom_weight_heatmap(path, compare_folder, save = False)
    plt.show()