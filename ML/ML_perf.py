import sys
sys.path.append('../') # parent folder: MastersThesis
from plot_set import *
from data_analysis import *
from use_network import *
import ast



def get_vmin_max(mat):
    p = 0.5
    # High scores = good
    vmin = np.nanmax((p*np.nanmax(mat), np.nanmin(mat)))
    vmax = np.nanmax(mat)
    return vmin, vmax

def staircase_perf(path, compare_folder = [], save = False):
    
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
    
   
    # # Get unique axis
    # D_axis =  list(set(D))
    # S_axis =  list(set(S))
    # shape = (len(S_axis), len(D_axis))
    
    # # Sort 
    # D = np.array(D); S = np.array(S)
    # S_sort = np.argsort(S)
    # D_sort = np.concatenate([np.argsort(D[S_sort][i:i+shape[1]])+i for i in range(0, len(D), shape[1])])
    # sort = S_sort[D_sort]
    
    # D = np.flip(D[sort].reshape(shape), 0)
    # S = np.flip(S[sort].reshape(shape), 0)
    # for key in P:  
    #     P[key] = np.flip(np.array(P[key])[sort].reshape(shape), 0)
    

 

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
    sns.heatmap(mat, xticklabels = D_axis, yticklabels = S_axis, cbar_kws={'label': 'Num. params'}, annot=True, fmt='.4g', vmin=None, vmax=None, ax=ax)
    ax.set_xlabel('Depth', fontsize=14)
    ax.set_ylabel('Start num. channels', fontsize=14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)



            
            
        
        
        
    

    if save:
        pass
        # plt.savefig('../article/figures/figure.pdf', bbox_inches='tight')


if __name__ == '__main__':
    # path = '../ML/staircase_1'
    path = '../ML/staircase_2'
    
    compare_folder = ['../Data/Baseline_fixmove/honeycomb/multi_stretch', '../Data/Baseline_fixmove/popup/multi_stretch']
    staircase_perf(path, compare_folder, save = False)
    plt.show()