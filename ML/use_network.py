from ML_utils import *
from networks import *

import sys
sys.path.append('../') # parent folder: MastersThesis
from plot_set import *

# from analysis.analysis_utils import *
from produce_figures.baseline_variables import *


def config_profile(model, config_path, stretch, F_N):
    model.eval() # Evaluation mode
    
    
    F_N_torch = torch.from_numpy(F_N.astype(np.float32))
    stretch_torch = torch.from_numpy(stretch.astype(np.float32))
    
    
    config = torch.from_numpy(np.load(config_path).astype(np.float32))
    config = config.unsqueeze(0).repeat(len(stretch_torch), 1, 1)
    vals = torch.stack((stretch_torch, F_N_torch), 1)

    output = model(config, vals).detach().numpy()
    rupture = output[:,-1] > 0.5
    
    # print(output[:, :])
    
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    plt.plot(stretch[~rupture], output[:,0][~rupture], 'o', markersize = 1.5, label = "No rupture")
    plt.plot(stretch[rupture], output[:,0][rupture], 'o', markersize = 1.5, label = "Rupture")
    
    plt.xlabel('Stretch', fontsize=14)
    plt.ylabel(r'$\langle F_\parallel \rangle$ [nN]', fontsize=14)
    plt.legend(fontsize = 13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    # plt.savefig('../article/figures/figure.pdf', bbox_inches='tight')

    
    
    

def compare_model_to_data(model, folder):
    
    colorbar_scale = [(0.1, 10), 'log']
    
    config_path = find_single_file(folder, '.npy')
    vars = ['data[\'stretch_pct\']', 'data[\'Ff\'][:, :, 0, 1]', 'data[\'F_N\']']
    axis_labels = [r'Stretch', r'$\langle F_\parallel \rangle$ [nN]', r'$F_N$ [nN]']
    fig, data = multi_plot_compare([folder], [config_path], vars, axis_labels, figsize = (7, 5), axis_scale = ['linear', 'linear'], colorbar_scale = colorbar_scale, equal_axes = [False, False], rupplot = True)
    axes = fig.axes
   
    stretch = data['stretch_pct']
    F_N = data['F_N']
    
  

    
    num_points = 100
    stretch_torch = torch.from_numpy(np.linspace(np.min(stretch), np.max(stretch), num_points).astype(np.float32))
    config = torch.from_numpy(np.load(config_path).astype(np.float32))
    config = config.unsqueeze(0).repeat(len(stretch_torch), 1, 1)
    
    model.eval()
    for k in F_N:
        F_N_torch = torch.from_numpy((np.full(len(stretch_torch), k)).astype(np.float32))
        vals = torch.stack((stretch_torch, F_N_torch), 1)

        output = model(config, vals).detach().numpy()
        rupture = output[:,-1] > 0.5


        color = get_color_value(k, colorbar_scale[0][0], colorbar_scale[0][1], scale = colorbar_scale[1], cmap = matplotlib.cm.viridis)
        axes[0].plot(stretch_torch, output[:, 0], color = color)
        
    
    
    
    


def load_model(weight_path):
    # model = VGGNet( mode = 0, 
    #                 input_num = 2, 
    #                 conv_layers = [(1, 16), (1, 32), (1, 64)], 
    #                 FC_layers = [(1, 512), (1,128)],
    #                 out_features = ['R', 'R', 'C'])
    # model = VGGNet( mode = 0, 
    #                 input_num = 2, 
    #                 conv_layers = [(1, 32), (1, 64), (1, 128)], 
    #                 FC_layers = [(1, 512), (1,128)],
    #                 out_features = ['R', 'R', 'R', 'R', 'R', 'C'])
    model = VGGNet( mode = 0, 
                    input_num = 2, 
                    conv_layers = [(1, 16), (1, 32)], 
                    FC_layers = [(1, 32), (1,16)],
                    out_features = ['R', 'R', 'R', 'R', 'R', 'C'])
    
    
    model = load_weights(model, weight_path)
    return model


def test_model_compare():
    model = load_model('training/C16C32D32D16_model_dict_state')

    # folder = '../Data/CONFIGS/honeycomb/hon_5' 
    # folder = '../Data/CONFIGS/popup/pop_4'
    
    # folder = '../Data/CONFIGS/honeycomb/hon_1' # hon3215 used in ML data
    folder = '../Data/Baseline_fixmove/honeycomb/multi_stretch' # hon3215 used in Baseline
    # folder = '../Data/CONFIGS/popup/pop_35' # pop1_7_5 used in ML data
    # folder = '../Data/Baseline_fixmove/popup/multi_stretch' # pop1_7_5 used in Baseline
    compare_model_to_data(model, folder)


def test_model_manual():
    model = load_model('training/test_model_dict_state')


    num_points = 100    
    config_path = '../config_builder/baseline/nocut.npy'
    # config_path = '../config_builder/baseline/pop1_7_5.npy'
    config_path = '../config_builder/baseline/hon3215.npy'
    stretch = np.linspace(0, 2, num_points)
    F_N = np.linspace(1, 1, num_points) # nN
    config_profile(model, config_path, stretch, F_N)



if __name__ == '__main__':
    # test_model_manual()
    test_model_compare()
    plt.show()
    
    # config_profile(config_path, model)
    