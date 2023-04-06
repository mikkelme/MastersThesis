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
        'max_epochs': 100,
        'max_file_num': None,
        'weight_decay': 0,
        'cyclic_lr': [20.0, 1e-3, 1e4], # [div_factor, max_lr, final_div_factor]
        'cyclic_momentum': [0.80, 0.90], # base_momentum, max_momentum
        'scheduler': None # [step size, factor], [10, 3]
    }

    model, criterion = best_model(mode = 0, batchnorm = True)[0] 
    # max_lr = [0.19098169151133398/2, 0.07099694630794687/2, 0.048523309416493646/2, 0.0078083987538332695/2]
   
   
    momentum = [0.99, 0.97, 0.95, 0.9]
    max_lr = [0.01803841387132259, 0.04496685997870468, 0.0114248692791861, 0.0019839040369852958]
    weight_decay = [1e-4, 1e-5, 1e-6, 0]    

    momentum_weight_search(model, criterion, data_root, ML_setting, max_lr, momentum, weight_decay, save_folder = 'mom_weight_search_short')