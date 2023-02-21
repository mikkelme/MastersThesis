# from RainforestDataset import *
# from utilities import *
# from Networks import *

from dataloaders import *
from ML_utils import *
from networks import *


def train_tmp(data_root, ML_setting, maxfilenum = None):
    """ Training...

    Args:
        data_root (string): root directory for data files
        ML_setting (dict): ML settings
        maxfilenum (int, optional): Maximum number of data points to include in the total dataset. Defaults to None.
    """    
    # config = get_config(use_gpu)
    # datasets, dataloaders = get_data(root_dir, config, maxfilenum)
    # device = get_device(config)
    
    # Data and device
    datasets, dataloaders = get_data(data_root, ML_setting, maxfilenum)
    device = get_device(ML_setting)
    
    model = LeNet(3) # TODO: Have as function input 

    # Model
   

    # Loss function, optimizer, scheduler
    criterion = nn.BCELoss() # TODO: This should be updated!!! XXX
    optimizer = optim.SGD(model.parameters(), lr = ML_setting['lr'], momentum = 0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR( optimizer, 
                                                    step_size = ML_setting['scheduler_stepsize'], 
                                                    gamma = ML_setting['scheduler_factor'], 
                                                    last_epoch = - 1, 
                                                    verbose = False)

    # Train and evaluate
    avgloss = train_epoch(model, dataloaders['train'], criterion, optimizer, device)
    
    
    
    exit()
    mode = 0 # full dataset
    best_epoch, best_avgprec, best_weights, best_scores, train_losses, test_losses, test_precs = train_and_evaluate(mode, dataloaders['train_rgb'], dataloaders['val_rgb'], model, criterion, optimizer, lr_scheduler, config['maxnumepochs'], device, config['numcl'] )


    print(f'Best epoch: {best_epoch}')
    print(f'Best avg precision: {best_avgprec}')
    # Save history, best model state and best model validation scores
    session_name = 'rgb'
    save_training_history(session_name, train_losses, test_losses, test_precs)
    save_best_model(session_name, model, best_weights)
    save_best_model_val_scores(session_name, best_scores, best_epoch)



if __name__=='__main__':
    # root_dir = '/itf-fi-ml/shared/IN5400/2022_mandatory1'
    # root_dir = '/Users/mikkelme/Documents/Github/IN5400_ML/Mandatory1/'
    
    data_root = '../data_pipeline/tmp_data'
    ML_setting = get_ML_setting()

    train_tmp(data_root, ML_setting, maxfilenum = None)
