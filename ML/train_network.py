# from RainforestDataset import *
# from utilities import *
# from Networks import *

from dataloaders import *
from ML_utils import *
from networks import *


def loss_func(outputs, labels):
    """ One version of a loss function """
    alpha = 0.8 # 1: priority of Fs MSE, 0: priority of rup BCE
    
    criterion = [nn.MSELoss(), nn.BCELoss()]
    
    not_ruptured = labels[:,1] == 0 # Only include Ff_loss when not ruptured
    if torch.all(~not_ruptured):
        Ff_loss = torch.tensor(0)
    else:
        Ff_loss = alpha*criterion[0](outputs[not_ruptured, 0], labels[not_ruptured, 0])

    rup_loss = (1-alpha)*criterion[1](outputs[:, 1], labels[:, 1])
    loss = Ff_loss + rup_loss

    return loss, Ff_loss, rup_loss


def train_tmp(data_root, model, ML_setting, save_best = None, maxfilenum = None):
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
    train_losses, validation_losses, best = train_and_evaluate(model, dataloaders, criterion, optimizer, lr_scheduler, ML_setting, device, save_best = save_best is not None)

    if save_best is not None:
        print(f'Best epoch: {best["epoch"]}')
        print(f'Best loss: {best["loss"]}')
        # save_training_history(name, train_losses, test_losses, test_precs)
        save_best_model(save_best, model, best['weights'])
        # save_best_model_val_scores(name, best_scores, best_epoch)
        
        

    # Fast plotting
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    plt.plot(train_losses[:, 0], label = "train loss")
    plt.plot(validation_losses[:, 0], label = "validation loss")
    plt.legend()
    
    # plt.figure(num=1, dpi=80, facecolor='w', edgecolor='k')
    # plt.plot(train_losses[:, 0], label = "train loss")
    # plt.plot(train_losses[:, 1], label = "train MSE")
    # plt.plot(train_losses[:, 2], label = "train BCE")
    # plt.legend()

    # plt.figure(num=2, dpi=80, facecolor='w', edgecolor='k')
    # plt.plot(validation_losses[:, 0], label = "validation loss")
    # plt.plot(validation_losses[:, 1], label = "validation MSE")
    # plt.plot(validation_losses[:, 2], label = "validation BCE")
    # plt.legend()

    plt.show()
    
    
    # print(f'Best epoch: {best_epoch}')
    # print(f'Best avg precision: {best_avgprec}')
    # # Save history, best model state and best model validation scores
    # session_name = 'rgb'
    # save_training_history(session_name, train_losses, test_losses, test_precs)
    # save_best_model(session_name, model, best_weights)
    # save_best_model_val_scores(session_name, best_scores, best_epoch)



if __name__=='__main__':
    # TODO: Initialize weights in model
    
    # data_root = '../data_pipeline/tmp_data'
    data_root = '../Data/ML/honeycomb'
    ML_setting = get_ML_setting()
    
    model = VGGNet(mode = 1)
   
    
    
    #
    #
    #
    # XXX Working here
    # TODO: abs errror gives nan, check that...
    #
    #
    
    train_tmp(data_root, model, ML_setting, save_best = './test1000', maxfilenum = 500)
