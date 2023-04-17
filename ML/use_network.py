import sys
sys.path.append('../') # parent folder: MastersThesis

# if 'MastersThesis' in sys.path[0]: # Local 
from produce_figures.baseline_variables import *
from ML.ML_utils import *
from ML.networks import *
from config_builder.build_config import *
    
# else: # Cluster
#     from baseline_variables import *
#     from ML_utils import *
#     from networks import *
#     from build_config import *
    

import ast
from plot_set import *
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
            assert False, 'Does this happen'
        else:
            image  = self.image.unsqueeze(0).repeat(len(stretch_torch), 1, 1) 
    
        vals = torch.stack((stretch_torch, F_N_torch), -1)

        # Get output
        self.output = self.model((image, vals)).detach().numpy()
        return image.detach().numpy(), vals.detach().numpy(), self.output


    def stretch_profile(self, stretch, F_N, ax = None):
        image, vals, output = self.predict(stretch, F_N)
        rupture = output[:,-1] > 0.5
    
        stretch = vals[:, 0]
        F_N = vals[:, 1]
        
        if ax is None:
            fig = plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
            ax = plt.gca()

    
        ax.plot(stretch[~rupture], output[:,0][~rupture], 'o', markersize = 1.5, label = "No rupture")
        ax.plot(stretch[rupture], output[:,0][rupture], 'o', markersize = 1.5, label = "Rupture")
        
        ax.set_xlabel('Stretch', fontsize=14)
        ax.set_ylabel(r'$\langle F_\parallel \rangle$ [nN]', fontsize=14)
        ax.legend(fontsize = 13)
        
        
        if ax is None:
            fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        # plt.savefig('../article/figures/figure.pdf', bbox_inches='tight')
        return ax

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
        Ff = data['Ff'][:, :, 0, 1]
    
        # Exact input for R2 calculation
        
    
        # Make ML input
        stretch_space = np.linspace(np.min(stretch), np.max(stretch), num_points)
        self.load_config(config_path)
        
        # Predict for different F_N
        for k in range(len(F_N)):
            
            # Get R2
            no_rupture = ~np.isnan(Ff[:, k]) 
            Ff_target = Ff[no_rupture, k]
            Ff_target_mean = np.mean(Ff_target)
            
            _, _, output = self.predict(stretch, F_N[k])
            Ff_pred = output[no_rupture, 0]
            SS_res = np.sum((Ff_pred - Ff_target)**2)
            SS_tot = np.sum((Ff_target - Ff_target_mean)**2)
            R2 = 1 - SS_res/SS_tot
            
            
            # Produce more smooth stretch curve fore plotting
            _, _, output = self.predict(stretch_space, F_N[k])
            no_rupture = output[:,-1] < 0.5
            color = get_color_value(F_N[k], colorbar_scale[0][0], colorbar_scale[0][1], scale = colorbar_scale[1], cmap = matplotlib.cm.viridis)
            # axes[0].plot(stretch_space, output[:, 0], color = color, label = f'R2 = {R2:g}')
            axes[0].plot(stretch_space[no_rupture], output[no_rupture, 0], color = color, label = f'R2 = {R2:g}')
        
        fig.legend(fontsize = 14)
    
    def evaluate_properties(self, stretch = np.linspace(0, 2, 100),  F_N = 5,  show = False):
        image, vals, output = self.predict(stretch, F_N)
        
        stretch = vals[:, 0]
        Ff = output[:, 0]
        rupture = output[:,-1] > 0.5
        
    
        
        # --- Property metrics
        metrics = {}
    
        # Practical rupture stretch    
        if np.any(rupture):
            prac_rup_stretch_idx = np.min(np.argwhere(rupture))
            if prac_rup_stretch_idx < 2: # If early rupture
                return None
                # metrics['Ff_min'] = (np.nan, np.nan)
                # metrics['Ff_max'] = (np.nan, np.nan)
                # metrics['Ff_max_diff'] = (np.nan, np.nan, np.nan)
                # metrics['Ff_max_drop'] = (np.nan, np.nan, np.nan)
                # return metrics
                    
        else:
            prac_rup_stretch_idx = -1
      
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
        metrics['Ff_max_diff'] = (stretch[Ffmin_idx], stretch[Ffmax_idx], Ff[Ffmax_idx]-Ff[Ffmin_idx])
        metrics['Ff_max_drop'] = (stretch[drop_start], stretch[drop_end], max_drop)
                 

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
    
    
    def get_feature_maps(self):
        model_children = list(self.model.children())
        module_list = model_children[0]
        
        
        
        conv_weights = []
        conv_part = []
        for module in module_list:
            if type(module) == nn.Linear: break
            conv_part.append(module)
            if type(module) == nn.Conv2d:
                conv_weights.append(module.weight)
        
        # XXX Hardcoded
        stretch_val = 0.5
        F_N_val = 5.0
        
        # Get input
        image = self.image.unsqueeze(0).repeat(1, 1, 1, 1)
        stretch = torch.tensor(stretch_val).view(1, 1, 1, 1).expand(-1, -1, image.shape[2], image.shape[3])
        F_N = torch.tensor(F_N_val).view(1, 1, 1, 1).expand(-1, -1, image.shape[2], image.shape[3])
        x = torch.cat((image, stretch, F_N), dim = 1)
      
      
      
        outputs = [] 
        for layer in conv_part:
            x = layer(x)
            if type(layer) == nn.Conv2d:
                outputs.append(x)
            
        print(len(outputs))
        print(len(conv_weights))
      
      
        # Plo feature maps
        shape = (4,4)
        for depth in reversed(range(len(outputs))):
            layer = outputs[depth]
            print(np.shape(layer))
            fig, axes = plt.subplots(*shape, num = unique_fignum(), figsize = (10,6))
            
            N = np.min((np.prod(np.shape(axes)), layer.shape[1]))
            for i in range(N):
                ax = axes[i//axes.shape[1], i%axes.shape[1]]
                ax.imshow(np.array(layer[0, i].detach()), cmap='viridis')
                ax.axis('off')
            fig.suptitle(f'Depth = {depth}')
            fig.supxlabel(r"$x$ (armchair direction)", fontsize = 14)
            fig.supylabel(r"$y$ (zigzag direction)", fontsize = 14)
            fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)        
            
        # TODO: Maybe plot on lines like shown here: 
        # https://towardsdatascience.com/convolutional-neural-network-feature-map-and-filter-visualization-f75012a5a49c
      
        plt.show()
      
        # atom_radii = 0.6
        # fig1 = plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k'); ax1 = fig1.gca()
        # plot_feature_map(test, ax1, atom_radii, cmap = 'gray', edgecolor = 'black') # Pattern   

     
    
    def explainable_AI_methods(self):        
        # TODO: Use some kind of linearization method to show which pixels 
        # it is most sensitive to in order to reveal information about attention behind prediciton 
        pass
    
        # TODO: Activation maps XXX
        
        
        
    
def test_model_manual(name = None):
    # Model 
    if name is None:
        name = 'training/USE'
    model_weights = os.path.join(name, 'model_dict_state')
    model_info = os.path.join(name, 'best_scores.txt')
    

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
    model_weights = os.path.join(name, 'model_dict_state')
    model_info = os.path.join(name, 'best_scores.txt')

    # Folder
    # folder = '../Data/CONFIGS/honeycomb/hon_5' 
    # folder = '../Data/CONFIGS/popup/pop_4'

    # folder = '../Data/CONFIGS/honeycomb/hon_1' # hon3215 used in ML data
    folder = '../Data/Baseline_fixmove/honeycomb/multi_stretch' # hon3215 used in Baseline
    # folder = '../Data/CONFIGS/popup/pop_35' # pop1_7_5 used in ML data
    # folder = '../Data/Baseline_fixmove/popup/multi_stretch' # pop1_7_5 used in Baseline

    # --- Compare --- #
    EV = Evaluater(model_weights, model_info)
    EV.compare_to_folder(folder)
    plt.show()
    
    
def show_CNN_layers(model_path):
    model_weights = os.path.join(name, 'model_dict_state')
    model_info = os.path.join(name, 'best_scores.txt')


    # folder = '../Data/CONFIGS/honeycomb/hon_1' # hon3215 used in ML data
    folder = '../Data/Baseline_fixmove/honeycomb/multi_stretch' # hon3215 used in Baseline
    # folder = '../Data/CONFIGS/popup/pop_35' # pop1_7_5 used in ML data
    # folder = '../Data/Baseline_fixmove/popup/multi_stretch' # pop1_7_5 used in Baseline
    # folder = '../Data/CONFIGS/RW/RW50'
    
    config_path = find_single_file(folder, '.npy')
    EV = Evaluater(model_weights, model_info)
    EV.load_config(config_path)
    EV.get_feature_maps() 
    
    

def plot_feature_map(mat, ax, radius, cmap, **param):
    full = build_graphene_sheet(np.ones(np.shape(mat)))
    
    pos = full.get_positions().reshape(*np.shape(mat), 3)
    on = mat > 0.5
    xmin = np.min(pos[on, 0])
    ymin = np.min(pos[on, 1])
    xmax = np.max(pos[on, 0])
    ymax = np.max(pos[on, 1])
    
    colors = get_color_value(mat, np.min(mat), np.max(mat), scale = 'linear', cmap=cmap)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            x, y = pos[i, j, 0:2]
            circle = plt.Circle((x, y), radius, facecolor = colors[i, j],  **param)
            ax.add_patch(circle)
                
    ax.axis('equal')
    
    return xmin, ymin, xmax, ymax



if __name__ == '__main__':
    
    # name = 'graphene_h_BN/C16C32C64D64D32D16'
    # name = 'training_1/C16C32D32D16'
    
    # folder = 'staircase_2'
    
    # name = f'{folder}/C8C16C32C64D32D16D8' 
    # name = f'{folder}/C8C16D16D8' 
    # name = f'{folder}/C16C16D16D16'
    # name = f'{folder}/C16C32C32D32D32D16' 
    # name = f'{folder}/C16C32C64C64D64D32D16'
    # name = f'{folder}/C16C32C64C64D512D128' 
    # name = f'{folder}/C16C32C64C128D64D32D16' 
    # name = f'{folder}/C16C32C64D64D32D16' # BEST
    # name = f'{folder}/C16C32C64D512D128' 
    # name = f'{folder}/C16C32D32D16'
    # name = f'{folder}/C32C64C128D128D64D32'
    
    
    # name = f'{folder}/S32D8'
    # name = f'{folder}/S4D14'
    
    # test_model_manual(name)
    # test_model_compare(name)
    # show_CNN_layers(name)
    # show_CNN_layers(name)
    
    
    # plt.show()
    # pass
    model_name = 'mom_weight_search_cyclic/m0w0/'
    model_weights = f'{model_name}/model_dict_state'
    model_info = f'{model_name}/best_scores.txt'
    EV = Evaluater(model_weights, model_info, config_path = '../config_builder/honeycomb/hon3131.npy')
    stretch = np.linspace(0, 2, 100)
    F_N = 5
    EV.stretch_profile(stretch, F_N)
    plt.show()
    
    
    # EV = Evaluater(model_weights, model_info, config_path = '../config_builder/baseline/hon3215.npy')
    # EV.load_config('../config_builder/baseline/pop1_7_5.npy')
    # EV.evaluate_properties(show = True)
    
    
    # stretch = np.linspace(0, 2, 100)
    # F_N = 10
    # # EV.stretch_profile(stretch, F_N)
    # EV.compare_to_folder(folder = '../Data/Baseline_fixmove/honeycomb/multi_stretch')
    
    # plt.show()