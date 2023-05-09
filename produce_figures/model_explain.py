import sys
sys.path.append('../') # parent folder: MastersThesis
import matplotlib.pyplot as plt
import numpy as np

from plot_set import *
from ML.use_network import *
from kirigami_patterns import *

def explain_predict(EV, save = False):
    F_N = 5
    stretch = np.linspace(0, 5, 300)

    fig = plt.figure(num = unique_fignum(), figsize = (10, 7), facecolor='w', edgecolor='k')
    num_cams = 6
    gs = fig.add_gridspec(3,num_cams)
    ax_curve = fig.add_subplot(gs[0, :-2])
    ax_conf = fig.add_subplot(gs[0, -2:])
    axes_cam_Ff = [fig.add_subplot(gs[1, i]) for i in range(num_cams)]
    axes_cam_rup = [fig.add_subplot(gs[2, i]) for i in range(num_cams)]
    
    # Find stretch limit
    IM, V, output = EV.predict(stretch, F_N)
    rup = output[:, -1] > 0.5
    stretch_lim = V[np.argmax(rup), 0] # What if not found? XXX
    
    # Predict for appropiate interval
    stretch = np.linspace(0, stretch_lim, len(stretch))
    EV.stretch_profile(stretch, F_N, ax_curve)
    
    indexes = np.round(np.linspace(0, len(stretch)-1, num_cams)).astype('int')
    ax_curve.set_xticks(stretch[indexes])
    # vline(ax_curve, stretch[idx], color = 'black', linestyle = '-', linewidth = .5, zorder = -1)
    # ax.set_title(f'Stretch = {stretch[idx]:0.2f}')
    for i, idx in enumerate(indexes):
        ax = axes_cam_Ff[i]
        EV.grad_cam(idx, 0, ax)
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax = axes_cam_rup[i]
        EV.grad_cam(idx, 5, ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'S = {stretch[idx]:0.4f}', pad = 11)

    axes_cam_Ff[0].set_ylabel("Grad. $F_f$")
    axes_cam_rup[0].set_ylabel("Grad. Rup.")
    
    # Plot kirigami configuration
    plot_sheet(IM[0], ax_conf, radius = 0.6, facecolor = 'grey', edgecolor = 'black', alpha = 1)
    ax_conf.grid(False)
    ax_conf.set_facecolor("white")
    ax_conf.set_xticks([])
    ax_conf.set_yticks([])
    ax_conf.set_xlabel(r"$x$ (armchair direction)", fontsize = 14)
    ax_conf.set_ylabel(r"$y$ (zigzag direction)", fontsize = 14)
    
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    if save is not False:
        fig.savefig(f'../article/figures/search/grad_cam_{save}.pdf', bbox_inches='tight')
        



if __name__ == '__main__':
    model_name = f'../ML/mom_weight_search_cyclic/m0w0/'
    model_weights = f'{model_name}/model_dict_state'
    model_info = f'{model_name}/best_scores.txt'

    # config_path = '../ML/GA_RN_start/top0.npy'; save = '_'.join(config_path.split('/')[-2:]).strip('.npy')
    # config_path = '../ML/RW_search/Ff_max_drop0_conf.npy'; save = 'RW_search_max_drop0'
    # EV = Evaluater(model_weights, model_info, config_path)
    
    
    # mat = pop_up(shape = (62, 106), size = (1,7), sp = 1, ref = (1, 4)); save = 'pop_1_7_1_1_4'
    # mat = pop_up(shape = (62, 106), size = (5,3), sp = 1, ref = (2, 5)); save = 'pop_5_3_1_2_5'
    mat = honeycomb(shape = (62, 106), xwidth = 5, ywidth = 3,  bridge_thickness = 5, bridge_len = 3, ref = (12,0)); save = 'hon_3_3_5_3_12_0'
    # mat = honeycomb(shape = (62, 106), xwidth = 3, ywidth = 3,  bridge_thickness = 3, bridge_len = 3, ref = (6,6)); save = 'hon_2_2_3_3_6_6'
    
    # builder = config_builder(mat)
    # builder.save_mat('../config_builder/test_set', save)
    # builder.save_view('../config_builder/test_set', 'sheet', save)
    # builder.view()
    
    
    
    EV = Evaluater(model_weights, model_info)
    EV.set_config(mat)
    # exit()
    
    
    # F_N = 5
    # stretch = np.linspace(0, 2, 500)
    # metrics = EV.evaluate_properties(stretch, F_N)
    # print(metrics)
    # EV.stretch_profile(stretch, F_N)
    # plt.show()
    
    
    explain_predict(EV, save)
    plt.show()