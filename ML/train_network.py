# from RainforestDataset import *
# from utilities import *
# from Networks import *

from dataloaders import *
from ML_utils import *
from networks import *


# def loss_func(outputs, labels):
#     """ One version of a loss function """
#     alpha = 0.5 # 1: priority of Fs MSE, 0: priority of rup BCE
    
#     criterion = [nn.MSELoss(), nn.BCELoss()]
    
#     not_ruptured = labels[:,1] == 0 # Only include Ff_loss when not ruptured
#     if torch.all(~not_ruptured):
#         Ff_loss = torch.tensor(0)
#     else:
#         Ff_loss = alpha*criterion[0](outputs[not_ruptured, 0], labels[not_ruptured, 0])

#     rup_loss = (1-alpha)*criterion[1](outputs[:, 1], labels[:, 1])
#     loss = Ff_loss + rup_loss

#     return loss, Ff_loss, rup_loss



#
# TODO: Working here
# Implement rupture_stretch as output to get better 
# reasoning of valid stretch range
#

def loss_func(outputs, labels):
    """ One version of a loss function """
    alpha = 0.5 # 1: priority of MSE, 0: priority of rup stretch MSE and rup BCE
    
    criterion = [nn.MSELoss(), nn.MSELoss(), nn.BCELoss()]
    
    not_ruptured = labels[:,1] == 0 # Only include Ff_loss when not ruptured
    if torch.all(~not_ruptured):
        Ff_loss = torch.tensor(0)
    else:
        Ff_loss = alpha*criterion[0](outputs[not_ruptured, 0], labels[not_ruptured, 0])

    rup_stretch_loss = 0.5*(1-alpha)*criterion[1](outputs[:, 1], labels[:, 1])
    is_ruptured_loss = 0.5*(1-alpha)*criterion[-1](outputs[:, 1], labels[:, 1])
    rup_loss = (rup_stretch_loss + is_ruptured_loss)/2
    
    loss = Ff_loss + rup_loss

    return loss, Ff_loss, rup_loss


def train(data_root, model, ML_setting, save_best = False, maxfilenum = None):
    """ Training...

    Args:
        data_root (string): root directory for data files
        ML_setting (dict): ML settings
        maxfilenum (int, optional): Maximum number of data points to include in the total dataset. Defaults to None.
    """    
    
    # Data and device
    datasets, dataloaders = get_data(data_root, ML_setting, maxfilenum)
    device = get_device(ML_setting)
    
   
    # Loss function, optimizer, scheduler
    criterion = loss_func
    optimizer = optim.SGD(model.parameters(), lr = ML_setting['lr'], momentum = 0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR( optimizer, 
                                                    step_size = ML_setting['scheduler_stepsize'], 
                                                    gamma = ML_setting['scheduler_factor'], 
                                                    last_epoch = - 1, 
                                                    verbose = False)



    # Train and evaluate
    train_val_hist, best = train_and_evaluate(model, dataloaders, criterion, optimizer, lr_scheduler, ML_setting, device, save_best = save_best is not None)
    if save_best is not False:      
        print(f'Best epoch: {best["epoch"]}')
        print(f'Best loss: {best["loss"]}')
        save_training_history(save_best, train_val_hist, ML_setting)
        save_best_model_scores(save_best, best, ML_setting)
        save_best_model(save_best, model, best['weights'])
        
        
        

    # Fast plotting
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    plt.plot(train_val_hist['train_loss_TOT'], label = "train loss")
    plt.plot(train_val_hist['val_loss_TOT'], label = "validation loss")
    plt.legend()
    plt.show()
    
    



if __name__=='__main__':
    data_root = ['../Data/ML_data/nocut', '../Data/ML_data/popup', '../Data/ML_data/honeycomb']
    ML_setting = get_ML_setting()
    
    # model = VGGNet(mode = 0)
    model = VGGNet(mode = 0, out_features = 2, conv_layers = [(1, 16), (1, 32), (1, 64)], FC_layers = [(1, 512), (1,128)])    
    train(data_root, model, ML_setting, save_best = 'test', maxfilenum = None)
