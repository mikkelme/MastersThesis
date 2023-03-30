from hypertuning import *


class A_staircase(Architectures):    
    """ For a given starting number of channels and depth
        add CNN layers with doubling number of channels
        and similary add FC with halfing number of nodes. """
        
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
        for s in start:
            for d in depth:
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
              
            
    
    
if __name__ == '__main__':
    # root = '../Data/ML_data/' # Relative (local)
    root = '/home/users/mikkelme/ML_data/' # Absolute path (cluster)
    data_root = [root+'baseline', root+'popup', root+'honeycomb', root+'RW']
    
    ML_setting = {
        'use_gpu': True,
        'lr': 0.01, 
        'batchsize_train': 32,
        'batchsize_val': 64,
        'max_epochs': 1000,
        'max_file_num': None,
        'scheduler_stepsize': 100,
        'scheduler_factor': 0.5
    }
    
    
    ########## Testing ############### 
    # root = '../Data/ML_data/' # Relative (local)
    # data_root = [root+'popup']

    # ML_setting = {
    #     'use_gpu': False,
    #     'lr': 0.01, 
    #     'batchsize_train': 32,
    #     'batchsize_val': 64,
    #     'max_epochs': 5,
    #     'max_file_num': None,
    #     'scheduler_stepsize': 200,
    #     'scheduler_factor': 0.5
    # }
    ################################### 
    
    
    A = A_staircase(mode = 0, batchnorm = True)
    train_architectures(A, data_root, ML_setting, save_folder = 'staircase_2')
    # print(A)
    # A.A = A.A[:2]
    
    
    
    
