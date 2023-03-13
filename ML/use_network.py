from ML_utils import *
from networks import *
import ast


import sys
sys.path.append('../') # parent folder: MastersThesis
from plot_set import *

# from analysis.analysis_utils import *
from produce_figures.baseline_variables import *

from scipy.signal import argrelextrema








class Evaluater():
    def __init__(self, model_weights, model_info, config_path = None):
        self.load_model(model_weights, model_info)
        
        if config_path is not None:
             self.load_config(config_path)
    
    def load_model(self, model_weights, model_info):
        """ Load model using info about class initialization and stored weights """
        
        # Get model settings from model_info path
        model_settings = {}
        infile = open(model_info, 'r')
        for line in infile:
            if 'Model settings' in line:
                for line in infile:
                    if '---' in line: break
                    key, val = line.strip('#\n ').split(' = ')
             
                    # Convert to right data type
                    if '[' in val or '(' in val:
                        if not ',' in val:
                            val = val.replace(' ', ', ')
                        val = ast.literal_eval(val)
                    else:
                        if val == 'True' or val == 'False':
                            val = bool(val)
                        else:
                            try:
                                val = int(val)
                            except ValueError:
                                pass # just string then
                    
                    model_settings[key] = val
        
        # Initialize model
        model = VGGNet(**model_settings)
        
        # Load weights
        self.model = load_weights(model, model_weights)
        self.model.eval() # Evaluation mode


    def load_config(self, config_path):
        self.image = torch.from_numpy(np.load(config_path).astype(np.float32))
        
    def set_config(self, mat):
        self.image = torch.from_numpy(mat.astype(np.float32))


    def predict(self, stretch, F_N):
        # Numpy array
        stretch = np.array(stretch)
        F_N = np.array(F_N)
        
        # Check sizes
        if F_N.size == 1: # Expand F_N
            F_N = np.full(np.shape(stretch), F_N)
        elif stretch.size == 1: # Expand stretch
            stretch = np.full(np.shape(F_N), stretch)
        else:
            assert stretch.size == F_N.size, f"stretch array of len {stretch.size} is not compatible with F_N of len {F_N.size}"

        # Torch tensor
        F_N_torch = torch.from_numpy(F_N.astype(np.float32))
        stretch_torch = torch.from_numpy(stretch.astype(np.float32))
        
        if len(stretch_torch.size()) == 0:
            image = self.image # .copy() XXX?
        else:
            image  = self.image.unsqueeze(0).repeat(len(stretch_torch), 1, 1) 
    
        vals = torch.stack((stretch_torch, F_N_torch), -1)

        # Get output
        self.output = self.model(image, vals).detach().numpy()
        return image.detach().numpy(), vals.detach().numpy(), self.output


    def stretch_profile(self, stretch, F_N):
        image, vals, output = self.predict(stretch, F_N)
        rupture = output[:,-1] > 0.5
    
        stretch = vals[:, 0]
        F_N = vals[:, 1]
    
        plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
        plt.plot(stretch[~rupture], output[:,0][~rupture], 'o', markersize = 1.5, label = "No rupture")
        plt.plot(stretch[rupture], output[:,0][rupture], 'o', markersize = 1.5, label = "Rupture")
        
        plt.xlabel('Stretch', fontsize=14)
        plt.ylabel(r'$\langle F_\parallel \rangle$ [nN]', fontsize=14)
        plt.legend(fontsize = 13)
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        # plt.savefig('../article/figures/figure.pdf', bbox_inches='tight')


    def compare_to_folder(self, folder, colorbar_scale = [(0.1, 10), 'log'], num_points = 100):
        # Plot comparison
        config_path = find_single_file(folder, '.npy')
        vars = ['data[\'stretch_pct\']', 'data[\'Ff\'][:, :, 0, 1]', 'data[\'F_N\']']
        axis_labels = [r'Stretch', r'$\langle F_\parallel \rangle$ [nN]', r'$F_N$ [nN]']
        fig, data = multi_plot_compare([folder], [config_path], vars, axis_labels, figsize = (7, 5), axis_scale = ['linear', 'linear'], colorbar_scale = colorbar_scale, equal_axes = [False, False], rupplot = True)
        axes = fig.axes
    
        # Get input data range
        stretch = data['stretch_pct']
        F_N = data['F_N']    
    
        # Make ML input
        stretch = np.linspace(np.min(stretch), np.max(stretch), num_points)
        self.load_config(config_path)
        
        # Predict for different F_N
        for k in F_N:
            _, _, output = self.predict(stretch, k)
            rupture = output[:,-1] > 0.5

            color = get_color_value(k, colorbar_scale[0][0], colorbar_scale[0][1], scale = colorbar_scale[1], cmap = matplotlib.cm.viridis)
            axes[0].plot(stretch, output[:, 0], color = color)
        
    
    def evaluate_properties(self, stretch = np.linspace(0, 2, 100),  F_N = 5,  show = False):
        
        # Input vals XXX Hardcoded for now XXX
        # num_points = 100
        # stretch = np.linspace(0, 2, num_points)
        # F_N = 5
        ##############################
        
        image, vals, output = self.predict(stretch, F_N)
        
        stretch = vals[:, 0]
        Ff = output[:, 0]
        rupture = output[:,-1] > 0.5
        
        # --- Property metrics
        metrics = {}
        # Practical rupture stretch
        prac_rup_stretch_idx = np.min(np.argwhere(rupture))
        # prac_rup_stretch = stretch[prac_rup_stretch_idx]
        
        # Min and max friction (before any rupture prediction)
        Ffmin_idx = np.argmin(Ff[:prac_rup_stretch_idx])
        Ffmax_idx = np.argmax(Ff[:prac_rup_stretch_idx])
        
        
        # Biggest forward drop in Ff
        loc_max = argrelextrema(Ff[:prac_rup_stretch_idx], np.greater_equal)[0]
        loc_min = argrelextrema(Ff[:prac_rup_stretch_idx], np.less_equal)[0]

        drop_start = 0; drop_end = 0; max_drop = 0
        for i in loc_max:
            for j in loc_min:
                if j > i: # Only look forward 
                    drop = Ff[i] - Ff[j]
                    if drop > max_drop:
                        drop_start = i
                        drop_end = j
                        max_drop = drop
                        
      
        metrics['prac_rup_stretch'] = stretch[prac_rup_stretch_idx]
        metrics['Ff_min'] = (stretch[Ffmin_idx], Ff[Ffmin_idx])
        metrics['Ff_max'] = (stretch[Ffmax_idx], Ff[Ffmax_idx])
        metrics['max_drop'] = (stretch[drop_start], stretch[drop_end], max_drop)
                 

        if show:
            plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
            plt.plot(stretch[~rupture], output[:,0][~rupture], 'o', markersize = 1.5, label = "No rupture")
            plt.plot(stretch[rupture], output[:,0][rupture], 'o', markersize = 1.5, label = "Rupture")
            # plt.plot(stretch[Ffmin_idx], Ff[Ffmin_idx], 'o')
            # plt.plot(stretch[Ffmax_idx], Ff[Ffmax_idx], 'o')
            if drop_end != 0:
                plt.plot(stretch[drop_start], Ff[drop_start], 'o')
                plt.plot(stretch[drop_end], Ff[drop_end], 'o')
                
            plt.show()    
        
        return metrics
    
def test_model_manual(name = None):
    # Model 
    if name is None:
        name = 'training/USE'

    model_weights = f'{name}_model_dict_state'
    model_info = f'{name}_best_scores.txt'

    # Config
    config_path = '../config_builder/baseline/nocut.npy'
    config_path = '../config_builder/baseline/pop1_7_5.npy'
    config_path = '../config_builder/baseline/hon3215.npy'

    # Input vals
    stretch = np.linspace(0, 2, 100)
    F_N = 5

    # --- Stretch profile --- #
    EV = Evaluater(model_weights, model_info, config_path)
    EV.stretch_profile(stretch, F_N)
    plt.show()


def test_model_compare(name = None):
    # Model 
    if name is None:
        name = 'training/USE'
    model_weights = f'{name}_model_dict_state'
    model_info = f'{name}_best_scores.txt'

    # Folder
    # folder = '../Data/CONFIGS/honeycomb/hon_5' 
    # folder = '../Data/CONFIGS/popup/pop_4'

    # folder = '../Data/CONFIGS/honeycomb/hon_1' # hon3215 used in ML data
    # folder = '../Data/Baseline_fixmove/honeycomb/multi_stretch' # hon3215 used in Baseline
    # folder = '../Data/CONFIGS/popup/pop_35' # pop1_7_5 used in ML data
    folder = '../Data/Baseline_fixmove/popup/multi_stretch' # pop1_7_5 used in Baseline

    # --- Compare --- #
    EV = Evaluater(model_weights, model_info)
    EV.compare_to_folder(folder)
    plt.show()
    




if __name__ == '__main__':
    
    name = 'graphene_h_BN/C16C32C64D64D32D16'
    # test_model_manual()
    test_model_compare(name)
    
    
    pass
    
    # model_weights = f'{name}_model_dict_state'
    # model_info = f'{name}_best_scores.txt'
    # EV = Evaluater(model_weights, model_info, config_path = '../config_builder/baseline/hon3215.npy')
    # EV.load_config('../config_builder/baseline/pop1_7_5.npy')
    # EV.evaluate_properties(show = True)
    
    
    # stretch = np.linspace(0, 2, 100)
    # F_N = 10
    # # EV.stretch_profile(stretch, F_N)
    # EV.compare_to_folder(folder = '../Data/Baseline_fixmove/honeycomb/multi_stretch')
    
    # plt.show()