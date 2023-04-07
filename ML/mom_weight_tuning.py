from hypertuning import *

    
if __name__ == '__main__':
    # root = '../Data/ML_data/' # Relative (local)
    root = '/home/users/mikkelme/ML_data/' # Absolute path (cluster)
    data_root = [root+'baseline', root+'popup', root+'honeycomb', root+'RW']
    
    
    ML_setting = {
        'use_gpu': torch.cuda.is_available(),
        'lr': 1e-4, 
        'batchsize_train': 32,
        'batchsize_val': 64,
        'max_epochs': 1000,
        'max_file_num': None,
        'weight_decay': 0,
        'momentum': 0.9,
        'cyclic_lr': [20.0, 1e-3, 1e4], # [div_factor, max_lr, final_div_factor]
        'cyclic_momentum': [0.80, 0.90], # base_momentum, max_momentum
        'scheduler': None # [step size, factor], [10, 3]
    }

    model, criterion = best_model(mode = 0, batchnorm = True)[0] 
    # max_lr = [0.19098169151133398/2, 0.07099694630794687/2, 0.048523309416493646/2, 0.0078083987538332695/2]
    # momentum = [0.99, 0.97, 0.95, 0.9]
    # max_lr = [0.01803841387132259, 0.04496685997870468, 0.0114248692791861, 0.0019839040369852958]
    # weight_decay = [1e-4, 1e-5, 1e-6, 0]    
   
   

    momentum = [0.85, 0.88, 0.91, 0.93, 0.95, 0.97, 0.99]
    weight_decay = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
   
   
    # --- Cyclic lr/momentum --- #
    lr_max = [0.0014631513629574113, 0.002310129700083158, 0.007236092872555868, 0.003935875227966555, 0.005336699231206312, 0.03578648210012845, 0.07099694630794687]
    momentum_weight_search(model, criterion, data_root, ML_setting, lr = lr_max, momentum = momentum, weight_decay = weight_decay, save_folder = 'mom_weight_search_cyclic', mode = 'cyclic_lr')


    # --- Constant lr/momentum --- #
    lr_sug = [5.543724670769404e-05, 5.982180380106274e-05, 5.982180380106274e-05, 6.9658674561388e-05, 0.00010998217754274403, 0.00012806723011028127, 0.00016092057297268975]
    # momentum_weight_search(model, criterion, data_root, ML_setting, lr = lr_sug, momentum = momentum, weight_decay = weight_decay, save_folder = 'mom_weight_search_constant', mode = 'constant_lr')
    