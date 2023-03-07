# from RainforestDataset import *
# from utilities import *
# from Networks import *

from dataloaders import *
from ML_utils import *
from networks import *


class Loss: # Maybe find better name: 'Criterion' or 'Loss_func'
    def __init__(self, alpha = [[1], [], [1/2, 1/2]], out_features = [['R'], [], ['R', 'C']]):
        """_summary_

        Args:
            alpha (lsit): Weight of input / category
            out_features (list of list): Describes output features associated to the categories 
                Friction, Other, Rupture. Defaults to [['R'], [], ['C']].
        """        
      
      
        assert len(out_features) == 3, f"out_features = {out_features} must have len 3, not {len(out_features)}." 
        self.alpha = [[], [], []]
        self.criterion = [[], [], []]
        self.cat_to_col_map = [[], [], []]
        
        idx_count = 0
        alpha_sum = 0
        for i, cat in enumerate(self.criterion):      
            for j, o in enumerate(out_features[i]):
                if hasattr(alpha[i], '__len__'):
                    self.alpha[i].append(alpha[i][j])
                    alpha_sum += alpha[i][j]
                else:
                    exit("This type of alpha list is not implemented yet")
                self.cat_to_col_map[i].append(idx_count)
                idx_count += 1
                if o == 'R': # Regression => MSE
                    cat.append(nn.MSELoss())
                elif o == 'C': # Classification => Binary cross entropy
                    cat.append(nn.BCELoss())
                else:
                    exit(f'out_feature {o} is not understood.')

            assert len(self.alpha[i]) == len(out_features[i]), f"Length of self.alpha does not match out_features at category i = {i}" 
            assert len(self.criterion[i]) == len(out_features[i]), f"Length of self.criterion does not match out_features at category i = {i}" 
            assert len(self.cat_to_col_map[i]) == len(out_features[i]), f"Length of self.cat_to_col_map does not match out_features at category i = {i}" 

        
        # Normalize weights in alpha:
        for i in range(len(self.alpha)):
            for j in range(len(self.alpha[i])):
                self.alpha[i][j] /= alpha_sum
        
    def __call__(self, outputs, labels):
        is_ruptured = labels[:,-1] # Convention to have is_ruptured last
        not_ruptured = is_ruptured == 0 
        
        loss = [0, 0, 0]
        
        # --- Ff loss --- #
        # Only include Ff_loss when not ruptured
        if not torch.all(~not_ruptured): 
            for i, crit in enumerate(self.criterion[0]):
                loss[0] += self.alpha[0][i] * crit(outputs[not_ruptured, self.cat_to_col_map[0][i]], labels[not_ruptured, self.cat_to_col_map[0][i]])
            if isinstance(loss[0], int):
                print("hep")
        # --- Other and rup loss --- #
        for k in range(1, len(loss)):
            for i, crit in enumerate(self.criterion[k]):
                loss[k] += self.alpha[k][i] * crit(outputs[:, self.cat_to_col_map[k][i]], labels[:, self.cat_to_col_map[k][i]])
            if isinstance(loss[k], int):
                loss[k] = torch.tensor(loss[k])
            
        tot_loss = loss[0] + loss[1] + loss[2]
        return tot_loss, loss[0], loss[1], loss[2]
            
            
        
# def loss_func(outputs, labels):
#     """ One version of a loss function """
#     alpha = 0.5 # 1: priority of MSE, 0: priority of rup stretch MSE and rup BCE
    
#     criterion = [nn.MSELoss(), nn.MSELoss(), nn.BCELoss()]
    
#     not_ruptured = labels[:,-1] == 0 # Only include Ff_loss when not ruptured
#     if torch.all(~not_ruptured):
#         Ff_loss = torch.tensor(0)
#     else:
#         Ff_loss = alpha*criterion[0](outputs[not_ruptured, 0], labels[not_ruptured, 0])

#     is_ruptured_loss = criterion[-1](outputs[:, -1], labels[:, -1])
    
#     if outputs.shape[1] > 2:
#         rup_stretch_loss = criterion[1](outputs[:, 1], labels[:, 1])
#         rup_loss = (rup_stretch_loss + is_ruptured_loss)/2
#     else:
#         rup_loss = is_ruptured_loss
    
#     loss = alpha*Ff_loss + (1-alpha)*rup_loss
    
#     return loss, Ff_loss, rup_loss


# def train(data_root, model, loss_func, ML_setting, save_best = False, maxfilenum = None):
#     """ Training...

#     Args:
#         data_root (string): root directory for data files
#         ML_setting (dict): ML settings
#         maxfilenum (int, optional): Maximum number of data points to include in the total dataset. Defaults to None.
#     """    
    
#     # Data and device
#     datasets, dataloaders = get_data(data_root, ML_setting, maxfilenum)
#     device = get_device(ML_setting)
    
   
#     # Loss function, optimizer, scheduler
#     # criterion = loss_func
#     optimizer = optim.SGD(model.parameters(), lr = ML_setting['lr'], momentum = 0.9)
    
#     if ML_setting['scheduler_stepsize'] is None or ML_setting['scheduler_factor'] is None:
#         lr_scheduler = None
#     else:
#         lr_scheduler = torch.optim.lr_scheduler.StepLR( optimizer, 
#                                                     step_size = ML_setting['scheduler_stepsize'], 
#                                                     gamma = ML_setting['scheduler_factor'], 
#                                                     last_epoch = - 1, 
#                                                     verbose = False)

#     if lr_scheduler is None:
#         exit(f'lr_scheduler is None | Not yet implemented')

#     # Train and evaluate
#     train_val_hist, best = train_and_evaluate(model, dataloaders, loss_func, optimizer, lr_scheduler, ML_setting, device, save_best = save_best is not None)
    
#     if save_best is not False and save_best is not None:      
#         print(f'Best epoch: {best["epoch"]}')
#         print(f'Best loss: {best["loss"]}')
#         save_training_history(save_best, train_val_hist, ML_setting)
#         save_best_model_scores(save_best, best, ML_setting)
#         save_best_model(save_best, model, best['weights'])
        
        
        

#     # Fast plotting
#     plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
#     plt.plot(train_val_hist['train_loss_TOT'], label = "train loss")
#     plt.plot(train_val_hist['val_loss_TOT'], label = "validation loss")
#     plt.legend()
#     plt.show()
    
    


class Trainer:
    def __init__(self, model, data_root, criterion, **ML_setting):
        self.model = model
        self.data_root = data_root
        self.criterion = criterion
        
        
        # Default settings
        self.ML_setting = {
            'use_gpu': False,
            'lr': 0.005,  # Learning rate
            'batchsize_train': 16,
            'batchsize_val': 64,
            'maxnumepochs': 35,
            'scheduler_stepsize': 10,
            'scheduler_factor': 0.3
        }
        
        self.ML_setting.update(ML_setting)
    
    
        self.optimizer = optim.SGD(model.parameters(), lr = self.ML_setting['lr'], momentum = 0.9)
        if self.ML_setting['scheduler_stepsize'] is None or self.ML_setting['scheduler_factor'] is None:
            self.lr_scheduler = None
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                        step_size = self.ML_setting['scheduler_stepsize'], 
                                                        gamma = self.ML_setting['scheduler_factor'], 
                                                        last_epoch = - 1, 
                                                        verbose = False)

        

    def train(self, save_best = False, maxfilenum = None):
        """ Training...

        Args:
            data_root (string): root directory for data files
            ML_setting (dict): ML settings
            maxfilenum (int, optional): Maximum number of data points to include in the total dataset. Defaults to None.
        """    

        # Data and device
        self.datasets, self.dataloaders = get_data(self.data_root, self.ML_setting, maxfilenum)
        self.device = get_device(ML_setting)

    
        if self.lr_scheduler is None:
            exit(f'lr_scheduler is None | Not yet implemented')

        # Train and evaluate
        train_val_hist, best = self.train_and_evaluate(save_best = save_best is not False)

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


    def train_and_evaluate(self, save_best = False):
        train_val_hist = {'epoch': [],
                        'train_loss_TOT': [],
                        'train_loss_MSE': [],
                        'train_loss_BCE': [],
                        'val_loss_TOT': [],
                        'val_loss_MSE': [],
                        'val_loss_BCE': []
                        }   
        
        best = {'epoch': -1, 'loss': 1e6, 'weights': None}
        num_epochs = self.ML_setting['maxnumepochs']

        print('Training model')
        for epoch in range(num_epochs):
            try:
                print('-' * 14)
                print(f'Epoch: {epoch+1}/{num_epochs}')


                # avgloss = train_epoch(model, dataloaders['train'], criterion, optimizer, device)
                avgloss = self.train_epoch()

                train_val_hist['epoch'].append(epoch)
                train_val_hist['train_loss_TOT'].append(avgloss[0])
                train_val_hist['train_loss_MSE'].append(avgloss[1])
                train_val_hist['train_loss_BCE'].append(avgloss[2])
                
                if self.lr_scheduler is not None: # TODO: Check this
                    self.lr_scheduler.step()

                avgloss, avg_metrics = self.evaluate_model()
                train_val_hist['val_loss_TOT'].append(avgloss[0])
                train_val_hist['val_loss_MSE'].append(avgloss[1])
                train_val_hist['val_loss_BCE'].append(avgloss[2])
                
                if epoch == 0:
                    for key in avg_metrics:
                        train_val_hist[key] = [avg_metrics[key]]
                else:
                    for key in avg_metrics:
                        train_val_hist[key].append(avg_metrics[key])
                        
                        
                
                # print(f'val_loss: {avgloss[0]:g}, Ff_abs: {avg_metrics["Ff_abs_error"]:g}, Ff_rel: {avg_metrics["Ff_rel_error"]:g} rup_acc: {avg_metrics["rup_acc"]:g}')                  
                print(f'val_loss: {avgloss[0]:g}, Ff_abs: {avg_metrics["Ff_abs_error"]:g}, Ff_rel: {avg_metrics["Ff_rel_error"]:g}, rup_stretch_abs: {avg_metrics["rup_stretch_abs_error"]:g}, rup_acc: {avg_metrics["rup_acc"]:g}')                  
                        
                
                if save_best:
                    if avgloss[0] < best['loss']: # TODO: Best criteria?
                        best['epoch'] = epoch
                        best['loss'] = avgloss[0]
                        best["Ff_abs_error"] =  avg_metrics["Ff_abs_error"]
                        best["Ff_rel_error"] = avg_metrics["Ff_rel_error"]
                        best["rup_stretch_abs_error"] = avg_metrics["rup_stretch_abs_error"]
                        best["rup_acc"] = avg_metrics["rup_acc"]                  
                        best['weights'] = self.model.state_dict()
                        
            
            
            except KeyboardInterrupt: break
        print('-' * 14)
        
        
        for key in train_val_hist:
            train_val_hist[key] = np.array(train_val_hist[key])
        
        
        return train_val_hist, best
        # return np.array(train_losses), np.array(validation_losses), best


    def common_things(self, data):
        # --- Evaluate --- #    
        image, vals = get_inputs(data, self.device)
        labels = get_labels(data, self.model.keys, self.device)
        outputs = self.model(image, vals)
        loss, Ff_loss, other_loss, rup_loss = self.criterion(outputs, labels)
        return loss, Ff_loss, other_loss, rup_loss, outputs, labels 
    
    

    def train_epoch(self):
        self.model.train() # Training mode
        dataloader = self.dataloaders['train']
        losses = []

        num_batches = len(dataloader)
        progress_bar_length = 8
        
        for batch_idx, data in enumerate(dataloader):
            # Zero gradients of all optimized torch.Tensor's
            self.optimizer.zero_grad() 

    
            loss, Ff_loss, other_loss, rup_loss, outputs, labels  = self.common_things(data)
        
        
            # --- Optimize --- #
            loss.backward()
            self.optimizer.step()
            losses.append([loss.item(), Ff_loss.item(), rup_loss.item()])

            # --- print progress --- #
            progress = int(((batch_idx+1)/num_batches)*progress_bar_length)
            print(f'\rLoss : {np.mean(losses):.4f} |{progress* "="}>{(progress_bar_length-progress)* " "}| {batch_idx+1}/{num_batches} ({100*(batch_idx+1)/num_batches:2.0f}%)', end = '')

        print()
        losses = np.array(losses)
        return np.mean(losses, axis = 0)


        
        

    def evaluate_model(self):
        self.model.eval() # Evaluation mode
        dataloader = self.dataloaders['val']
        
        
        with torch.no_grad():
            losses = []
            Ff_abs_error = []
            Ff_rel_error = []
            rup_stretch_abs_error = []
            accuracy = []
            
            for batch_idx, data in enumerate(dataloader):
                # --- Evaluate --- #
                loss, Ff_loss, other_loss, rup_loss, outputs, labels  = self.common_things(data)
                
                
                
        
                # --- Analyse --- #
                losses.append([loss.item(), Ff_loss.item(), rup_loss.item()])
                non_rupture = labels[:, -1] < 0.5
                
                # Additional metrics
                Ff_diff = outputs[non_rupture, 0] - labels[non_rupture, 0]
                Ff_abs_diff = torch.mean(torch.abs(Ff_diff))
                Ff_rel_diff = torch.mean(torch.abs(Ff_diff/labels[non_rupture, 0]))
                
                rup_stretch_diff = outputs[non_rupture, 1] - labels[non_rupture, 1]
                rup_stretch_abs_diff = torch.mean(torch.abs(rup_stretch_diff))
                
                rup_pred = torch.round(outputs[:,-1])
                acc = torch.sum(rup_pred == labels[:,-1])/len(rup_pred)
                
                
                # Add to list
                Ff_abs_error.append(Ff_abs_diff.item())
                Ff_rel_error.append(Ff_rel_diff.item())
                rup_stretch_abs_error.append(rup_stretch_abs_diff.item())
                accuracy.append(acc.item())
                
                
            losses = np.array(losses)
            avg_metrics = {
                            'Ff_abs_error': np.mean(np.array(Ff_abs_error)), 
                            'Ff_rel_error': np.mean(np.array(Ff_rel_error)),
                            'rup_stretch_abs_error': np.mean(np.array(rup_stretch_abs_error)),
                            'rup_acc': np.mean(np.array(accuracy)) 
                        }
            
            
            return np.mean(losses, axis = 0), avg_metrics
            # return np.mean(losses, axis = 0), np.mean(abs_error), np.mean(accuracy)



                

    


if __name__=='__main__':
    data_root = ['../Data/ML_data/baseline', '../Data/ML_data/popup', '../Data/ML_data/honeycomb']
    ML_setting = get_ML_setting()
    
    # model = VGGNet(mode = 0)
    model = VGGNet( mode = 0, 
                    input_num = 2, 
                    conv_layers = [(1, 16), (1, 32), (1, 64)], 
                    FC_layers = [(1, 512), (1,128)],
                    out_features = ['R', 'R', 'C'],
                    keys = ['Ff_mean', 'rupture_stretch', 'is_ruptured'])
    

    criterion = Loss(alpha = [[1], [], [1/2, 1/2]], out_features = [['R'], [], ['R', 'C']])
    
    coach = Trainer(model, data_root, criterion)
    coach.train(save_best = False, maxfilenum = 500)
    # coach = Trainer(model, data_root, loss, **ML_setting)
    # coach = Trainer(model, data_root, loss, use_gpu = True)
    
    
    # train(data_root, model, loss, ML_setting, save_best = 'training/more_output', maxfilenum = 500)
