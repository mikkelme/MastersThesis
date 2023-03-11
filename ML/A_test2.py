from hypertuning import *
from A_test1 import *

    
if __name__ == '__main__':
     # root = '../Data/ML_data/' # Relative
    root = '/home/users/mikkelme/ML_data/' # Absolute cluster
    data_root = [root+'baseline', root+'popup', root+'honeycomb']
    
    ML_setting = {
        'use_gpu': True,
        'lr': 0.001,  # Learning rate
        'batchsize_train': 32,
        'batchsize_val': 64,
        'max_epochs': 300,
        'max_file_num': None,
        'scheduler_stepsize': None, # 10
        'scheduler_factor': None # 0.3
    }
    
    
    A = A_test(mode = 0, batchnorm = True)
    train_architectures(A, data_root, ML_setting, save_folder = 'ghBN_3')
    
    
    