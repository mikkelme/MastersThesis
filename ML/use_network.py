from ML_utils import *
from networks import *

import sys
sys.path.append('../') # parent folder: MastersThesis
from plot_set import *


def config_profile(model, config_path, stretch, F_N):
    model.eval() # Evaluation mode
    
    
    F_N_torch = torch.from_numpy(F_N.astype(np.float32))
    stretch_torch = torch.from_numpy(stretch.astype(np.float32))
    
    
    config = torch.from_numpy(np.load(config_path).astype(np.float32))
    config = config.unsqueeze(0).repeat(len(stretch_torch), 1, 1)
    vals = torch.stack((stretch_torch, F_N_torch), 1)

    output = model(config, vals).detach().numpy()
    rupture = output[:,-1] > 0.5
    
    print(output[:, 1])
    
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    plt.plot(stretch[~rupture], output[:,0][~rupture], 'o', markersize = 1, label = "No rupture")
    plt.plot(stretch[rupture], output[:,0][rupture], 'o', markersize = 1, label = "Rupture")
    
    plt.xlabel('Stretch', fontsize=14)
    plt.ylabel(r'$\langle F_\parallel \rangle$ [nN]', fontsize=14)
    plt.legend(fontsize = 13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    # plt.savefig('../article/figures/figure.pdf', bbox_inches='tight')

    
    
    

def compare_model():
    pass


def load_model(weight_path):
    # model = VGGNet(mode = 1)
    model = VGGNet(mode = 0, out_features = 3, conv_layers = [(1, 16), (1, 32), (1, 64)], FC_layers = [(1, 512), (1,128)])
    
    model = load_weights(model, weight_path)
    return model


def test_model():
    model = load_model('test_model_dict_state')
    num_points = 100
    
    # config_path = '../config_builder/baseline/nocut.npy'
    config_path = '../config_builder/baseline/pop1_7_5.npy'
    # config_path = '../config_builder/baseline/hon3215.npy'
    stretch = np.linspace(0, 2, num_points)
    F_N = np.linspace(1, 1, num_points) # nN
    config_profile(model, config_path, stretch, F_N)



if __name__ == '__main__':
    test_model()
    plt.show()
    
    
    # config_profile(config_path, model)
    