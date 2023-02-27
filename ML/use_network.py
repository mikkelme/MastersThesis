from ML_utils import *
from networks import *

import sys
sys.path.append('../') # parent folder: MastersThesis
from plot_set import *


def config_profile(model):
    model.eval() # Evaluation mode
    
    F_N_val = 1e-9
    stretch_space = np.linspace(0, 2, 100)
    config_path = '../config_builder/baseline/hon3215.npy'
    
    
    FN = torch.from_numpy(np.full(len(stretch_space), F_N_val, dtype = np.float32))
    stretch = torch.from_numpy(np.array(stretch_space, dtype = np.float32))
    
    
    config = torch.from_numpy(np.load(config_path).astype(np.float32))
    config = config.unsqueeze(0).repeat(len(stretch_space), 1, 1)
    vals = torch.stack((stretch, FN), 1)

    output = model(config, vals).detach().numpy()
    rupture = output[:,1] > 0.5
    
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    # output[:,0]*1e9 # To nN units
    plt.plot(stretch_space[~rupture], output[:,0][~rupture], label = "No rupture")
    plt.plot(stretch_space[rupture], output[:,0][rupture], label = "Rupture")
    
    plt.xlabel('Stretch', fontsize=14)
    plt.ylabel(r'$\langle F_\parallel \rangle$ [nN]', fontsize=14)
    plt.legend(fontsize = 13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    # plt.savefig('../article/figures/figure.pdf', bbox_inches='tight')

    print(output)
    
    plt.show()


def load_model(weight_path):
    model = VGGNet(mode = 1)
    model = load_weights(model, weight_path)
    return model

if __name__ == '__main__':
    model = load_model('test1000_model_dict_state')
    
    config_profile(model)
    