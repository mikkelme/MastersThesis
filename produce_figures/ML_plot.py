import sys
sys.path.append('../') # parent folder: MastersThesis
import matplotlib.pyplot as plt
import numpy as np

from ML.hypertuning import *



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
        
        start_depth = [(2,6), (4,8), (8,6), (16,6)]
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
    
    ML_setting = {
        'use_gpu': False,
        'lr': 0.0001,
        'batchsize_train': 32,
        'batchsize_val': 64,
        'max_epochs': 1000,
        'max_file_num': None,
        'scheduler_stepsize': None, #100,
        'scheduler_factor': None #0.5
    }


    
    fig = plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
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
        
        # Cut end until below threshold
        # cut = np.argwhere(loss < 1)[-1]
        # map = np.logical_and(1e-5 < lr, lr < lr[cut]) 
        
        map = 1e-5 < lr
        
        plt.plot(lr[map], loss[map], color = color_cycle(i), label = f'{model.name} ({num_params:1.2e})')
        plt.plot(lr[sug_idx], loss[sug_idx], color = color_cycle(i), marker = 'o')
        plt.xscale('log')
        
        plt.xlabel('Learning rate', fontsize=14)
        plt.ylabel('Loss', fontsize=14)

        break
        
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.legend(fontsize = 13)
    
    if save:
        plt.savefig("../article/figures/ML/LR_range_specific.pdf", bbox_inches="tight")

    
    
def LR_range_full(filename):
    # TODO: Make LR vs param plot XXX
        
    pass



if __name__ == '__main__':
    
    LR_range_specific(A_staircase_subset(mode = 0, batchnorm = True), save = False)
    LR_range_full(filename)
    
    plt.show()
