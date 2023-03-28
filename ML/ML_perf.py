import sys
sys.path.append('../') # parent folder: MastersThesis
from plot_set import *
from data_analysis import *
import ast



def staircase_perf(path, save = False):
    
    folders = os.listdir(path)
    
    S = [] # Start
    D = [] # Depth
    P = {}
    for folder in folders:
        if '.DS_Store' in folder:
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
                

    # --- Organize into matrix (D, S, P) --- #
    # Get unique axis
    D_axis =  list(set(D))
    S_axis =  list(set(S))
    shape = (len(S_axis), len(D_axis))
    
    # Sort 
    D = np.array(D); S = np.array(S)
    S_sort = np.argsort(S)
    D_sort = np.concatenate([np.argsort(D[S_sort][i:i+shape[1]])+i for i in range(0, len(D), shape[1])])
    sort = S_sort[D_sort]
    
    D = np.flip(D[sort].reshape(shape), 0)
    S = np.flip(S[sort].reshape(shape), 0)
    for key in P:  
        P[key] = np.flip(np.array(P[key])[sort].reshape(shape), 0)
    

    # --- Plotting --- #
    # R2
    fig, ax = plt.subplots(num = unique_fignum(), figsize=(10, 6))
    mat = P['R2_0']
    vmin = np.max((0.96, np.min(mat)))
    vmax = np.max(mat)
    sns.heatmap(mat, xticklabels = D[0, :], yticklabels = S[:, 0], cbar_kws={'label': r'$R_2$ $\langle F_{\parallel} \rangle$ '}, annot=True, fmt='.4g', vmin=vmin, vmax=vmax, ax=ax)
    ax.set_xlabel('Depth', fontsize=14)
    ax.set_ylabel('Start num. channels', fontsize=14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)

    # Loss
    fig, ax = plt.subplots(num = unique_fignum(),  figsize=(10, 6))
    mat = P['val_tot_loss']
    vmin = np.min(mat)
    vmax = np.min((0.03, np.max(mat)))
    sns.heatmap(mat, xticklabels = D[0, :], yticklabels = S[:, 0], cbar_kws={'label': 'Validation loss'}, annot=True, fmt='.4g', vmin=vmin, vmax=vmax, ax=ax)
    ax.set_xlabel('Depth', fontsize=14)
    ax.set_ylabel('Start num. channels', fontsize=14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    # Epoch
    fig, ax = plt.subplots(num = unique_fignum(),  figsize=(10, 6))
    mat = P['epoch']
    sns.heatmap(mat, xticklabels = D[0, :], yticklabels = S[:, 0], cbar_kws={'label': 'Epoch'}, annot=True, fmt='.4g', vmin=None, vmax=None, ax=ax)
    ax.set_xlabel('Depth', fontsize=14)
    ax.set_ylabel('Start num. channels', fontsize=14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    # Num. Params
    fig, ax = plt.subplots(num = unique_fignum(),  figsize=(10, 6))
    mat = P['num_params']
    sns.heatmap(mat, xticklabels = D[0, :], yticklabels = S[:, 0], cbar_kws={'label': 'Num. params'}, annot=True, fmt='.4g', vmin=None, vmax=None, ax=ax)
    ax.set_xlabel('Depth', fontsize=14)
    ax.set_ylabel('Start num. channels', fontsize=14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)



    # Evaluation on selected configurations
    compare_folder = ['../Data/Baseline_fixmove/honeycomb/multi_stretch']
    # '../Data/Baseline_fixmove/popup/multi_stretch'
    for folder in compare_folder:
        model_weights = os.path.join(path, 'model_dict_state')
        model_info = os.path.join(path, 'best_scores.txt')
        EV = Evaluater(model_weights, model_info)
        
        read_multi_folder(folder):
        
        
        EV.compare_to_folder(folder)
    

    if save:
        pass
        # plt.savefig('../article/figures/figure.pdf', bbox_inches='tight')


if __name__ == '__main__':
    path = '../ML/staircase_1'
    staircase_perf(path)
    plt.show()