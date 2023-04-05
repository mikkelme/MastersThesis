import sys
sys.path.append('../') # parent folder: MastersThesis

if 'MastersThesis' in sys.path[0]: # Local 
    from ML.dataloaders import *
    from ML.ML_utils import *
    from ML.networks import *
else: # Cluster
    from dataloaders import *
    from ML_utils import *
    from networks import *
    

from collections import OrderedDict

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
        self.out_features = out_features
        self.criterion = [[], [], []]
        self.cat_to_col_map = [[], [], []]
        
        
        self.num_out_features = 0
        alpha_sum = 0
        for i, cat in enumerate(self.criterion):      
            for j, o in enumerate(out_features[i]):
                if hasattr(alpha[i], '__len__'):
                    self.alpha[i].append(alpha[i][j])
                    alpha_sum += alpha[i][j]
                else:
                    exit("This type of alpha list is not implemented yet")
                self.cat_to_col_map[i].append(self.num_out_features)
                self.num_out_features += 1
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
                exit("Why did I write af if-statement for this?")
                
        # --- Other and rup loss --- #
        for k in range(1, len(loss)):
            for i, crit in enumerate(self.criterion[k]):
                loss[k] += self.alpha[k][i] * crit(outputs[:, self.cat_to_col_map[k][i]], labels[:, self.cat_to_col_map[k][i]])
            if isinstance(loss[k], int):
                loss[k] = torch.tensor(loss[k])
            
        
        tot_loss = loss[0] + loss[1] + loss[2]
        return tot_loss, loss[0], loss[1], loss[2]
            
        


class Trainer:
    def __init__(self, model, data_root, criterion, **ML_setting):
        # Default ML settings
        self.ML_setting = {
            'use_gpu': False,
            'lr': 0.005,  # Learning rate
            'batchsize_train': 16,
            'batchsize_val': 64,
            'max_epochs': 100,
            'max_file_num': None,
            'scheduler_stepsize': 10,
            'scheduler_factor': 0.3
        }
        # Update 
        self.ML_setting.update(ML_setting)
        

        self.device = get_device(self.ML_setting)
        self.model = model.to(self.device)
        self.data_root = data_root
        self.criterion = criterion
        
       
        
        # Check that model and criterion is build for the same number number and style of output features
        model_out_feat = self.model.out_features
        criterion_out_feat = [item for sublist in self.criterion.out_features for item in sublist]        
        assert len(model_out_feat) == len(criterion_out_feat), f"Model expected {len(model_out_feat)} output features: {self.model.keys}, but criterion was build for {self.criterion.num_out_features} of style: {self.criterion.out_features}."
        assert np.all(np.array(model_out_feat) == np.array(criterion_out_feat)), f'Model output feature style: {model_out_feat} is not equal to that of the criteiron {criterion_out_feat}'
        
        Ff_style_keys = ['Ff_mean', 'Ff_max', 'Ff_mean_std', 'contact', 'contact_std']
        non_Ff_style_keys = self.model.keys[len(self.criterion.out_features[0]): ]
        
        # Check that none Ff_style keys was misplaced
        for key in non_Ff_style_keys:
            if key in Ff_style_keys:
                print(f"Key: \"{key}\" was used as non Ff_style key.")
                print(f'Keys: {Ff_style_keys}, must be used as Ff_style variable')
                exit()
        
        
    
        self.optimizer = optim.Adam(model.parameters(), lr = self.ML_setting['lr'], weight_decay = self.ML_setting['weight_decay'])      
        self.history = OrderedDict([('epoch', []), ('train_loss', []), ('val_loss', [])])
             

    def learn(self, max_epochs = None, max_file_num = None):
        """ Train the model """
        
        if max_epochs is not None:
            self.ML_setting['max_epochs'] = max_epochs
            
        if max_file_num is not None:
            self.ML_setting['max_file_num'] = max_file_num
        
        
        # Data 
        self.datasets, self.dataloaders = get_data(self.data_root, self.ML_setting)
        
        
        
             
        
        if self.ML_setting['scheduler'] is None:
            self.lr_scheduler = None
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                        step_size = self.ML_setting['scheduler'][0], 
                                                        gamma = self.ML_setting['scheduler'][1], 
                                                        last_epoch = - 1, 
                                                        verbose = False)
            
        
        if self.ML_setting['cyclic_lr'] is None:
            self.cyclic_lr = none
        else:
            
            self.cyclic_lr = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                max_lr = self.ML_setting['cyclic_lr'][1],
                                                total_steps=None, 
                                                epochs = self.ML_setting['max_epochs'], 
                                                steps_per_epoch=len(self.dataloaders['train']), 
                                                pct_start=0.3, 
                                                anneal_strategy='cos', 
                                                cycle_momentum=True, 
                                                base_momentum = self.ML_setting['cyclic_momentum'][0], 
                                                max_momentum = self.ML_setting['cyclic_momentum'][1], 
                                                div_factor = self.ML_setting['cyclic_lr'][0], 
                                                final_div_factor = self.ML_setting['cyclic_lr'][2])
            
            

        
        # --- Pre analyse validation data to get SS_tot --- #
        # Get mean
        self.num_out_features = len(self.model.out_features)
        val_sum = torch.zeros(self.num_out_features).to(self.device)
        val_num = torch.zeros(self.num_out_features).to(self.device)
        
        for batch_idx, data in enumerate(self.dataloaders['val']):
            labels = get_labels(data, self.model.keys, self.device)
            non_ruptured = labels[:, -1] < 0.5
            for i in range(labels.shape[1]):
                not_nan = ~torch.isnan(labels[:, i])
                d = labels[:, i]
                val_sum[i] += torch.sum(d[not_nan])
                val_num[i] += len(d[not_nan])
        
        
        val_mean = val_sum / val_num
        
        # Get SS_tot
        self.val_SS_tot = torch.zeros(self.num_out_features).to(self.device)
        for batch_idx, data in enumerate(self.dataloaders['val']):
            labels = get_labels(data, self.model.keys, self.device)
            non_ruptured = labels[:, -1] < 0.5
            
            for i in range(labels.shape[1]):
                not_nan = ~torch.isnan(labels[:, i])
                d = labels[not_nan, i]
                self.val_SS_tot[i] += torch.sum((d - val_mean[i])**2)


        # --- Train and evaluate --- #
        self.train_and_evaluate()
        
        # Print best
        print(f'Best epoch: {self.best["epoch"]}')
        print(f'Best val_loss: {self.best["loss"]}')

        # --- Clean up history --- #
        # Unpack train_ and val_loss
        train_loss = self.history['train_loss']
        self.history.pop('train_loss')
        self.history['train_tot_loss'] = train_loss[:, 0]
        self.history['train_Ff_loss'] = train_loss[:, 1]
        self.history['train_other_loss'] = train_loss[:, 2]
        self.history['train_rup_loss'] = train_loss[:, 3]
        
        val_loss = self.history['val_loss']
        self.history.pop('val_loss')
        self.history['val_tot_loss'] = val_loss[:, 0]
        self.history['val_Ff_loss'] = val_loss[:, 1]
        self.history['val_other_loss'] = val_loss[:, 2]
        self.history['val_rup_loss'] = val_loss[:, 3]
        
        
        # Reorder a bit for txt file
        self.history.move_to_end('train_rup_loss', last = False)
        self.history.move_to_end('train_other_loss', last = False)
        self.history.move_to_end('train_Ff_loss', last = False)
        self.history.move_to_end('train_tot_loss', last = False)
        
        self.history.move_to_end('val_rup_loss', last = False)
        self.history.move_to_end('val_other_loss', last = False)
        self.history.move_to_end('val_Ff_loss', last = False)
        self.history.move_to_end('val_tot_loss', last = False)
            
        self.history.move_to_end('epoch', last = False)
        

    def train_and_evaluate(self):
        best = {'epoch': -1, 'loss': 1e6, 'weights': None}
        num_epochs = self.ML_setting['max_epochs']


        # Add keys to history
        # The divided for-loops gives better order
        for i in range(self.num_out_features):
            self.history[f'R2_{i}'] = []
        for i in range(self.num_out_features):
            self.history[f'abs_{i}'] = []
        for i in range(self.num_out_features):
            self.history[f'rel_{i}'] = []
        self.history[f'acc'] = []
        self.history[f'lr'] = []

        print('Training model')
        for epoch in range(num_epochs):
            try:
                print('-' * 14)
                print(f'Epoch: {epoch+1}/{num_epochs}')

                # --- Train --- #
                train_loss = self.train_epoch()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # --- Validate --- #
                val_loss, R2, abs_error, rel_error, accuracy = self.evaluate_model()
            
                # --- Store information -- #
                # Add to history
                self.history['epoch'].append(epoch)
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                
                for i in range(self.num_out_features):
                    self.history[f'R2_{i}'].append(R2[i])
                    self.history[f'abs_{i}'].append(abs_error[i])
                    self.history[f'rel_{i}'].append(rel_error[i])
                self.history[f'acc'].append(accuracy)
                self.history[f'lr'].append(self.optimizer.param_groups[-1]['lr'])
                
                
                # Print to terminal 
                print(f'Val_loss: {val_loss[0]:g}')
                print('R2:', [f'{val:g}' for i, val in enumerate(R2)])
                print('Abs:', [f'{val:g}' for i, val in enumerate(abs_error)])
                print('Rel:', [f'{val:g}' for i, val in enumerate(rel_error)])
                print(f'Acc: {accuracy}')
                
                
                # --- Save best --- #
                if best['epoch'] < 0 or val_loss[0] < best['loss']:
                    best['epoch'] = epoch
                    best['loss'] = val_loss[0]
                    best['weights'] = self.model.state_dict()
                        
            
            except KeyboardInterrupt: break
        print('-' * 14)
        print("Training done")
        
        for key in self.history:
            self.history[key] = np.array(self.history[key])
        
        self.best = best

    def train_epoch(self):
        self.model.train() # Training mode
        dataloader = self.dataloaders['train']
        num_batches = len(dataloader)
        losses = np.zeros((num_batches, 4))
        
        progress_bar_length = 8
        for batch_idx, data in enumerate(dataloader):
            self.optimizer.zero_grad() # Zero gradients of all optimized torch.Tensor's

            # --- Forward pass --- #
            loss, Ff_loss, other_loss, rup_loss, outputs, labels  = self.forward_pass(data)
        
            # --- Optimize --- #
            loss.backward()
            self.optimizer.step()
            
            # --- Collect losses --- #
            losses[batch_idx] = loss.item(), Ff_loss.item(), other_loss.item(), rup_loss.item()
            mean = np.mean(losses[:batch_idx+1, 0])

            # --- print progress --- #
            progress = int(((batch_idx+1)/num_batches)*progress_bar_length)
            print(f'\rTrain loss : {np.mean(losses[:batch_idx+1, 0]):.4f} |{progress* "="}>{(progress_bar_length-progress)* " "}| {batch_idx+1}/{num_batches} ({100*(batch_idx+1)/num_batches:2.0f}%)', end = '')
            
            if self.cyclic_lr is not None:
                self.cyclic_lr.step()


        print()
        return np.mean(losses, axis = 0)
                

    def evaluate_model(self):
        self.model.eval() # Evaluation mode
        dataloader = self.dataloaders['val']
        
        
        with torch.no_grad():
            losses = []
            
            abs_error_list = []
            rel_error_list = []
            accuracy_list = []
            
        
            val_SS_res = torch.zeros(self.num_out_features).to(self.device)
            for batch_idx, data in enumerate(dataloader):
                # --- Forward pass --- #
                loss, Ff_loss, other_loss, rup_loss, outputs, labels  = self.forward_pass(data)
                
                # --- Analyse --- #
                losses.append([loss.item(), Ff_loss.item(), other_loss.item(), rup_loss.item()])
                non_rupture = labels[:, -1] < 0.5
                non_nan = ~torch.isnan(labels)
                
                
            
                
                # Additional metrics
                output_diff = torch.where(non_nan, outputs - labels, 0)
                abs_error = torch.abs(output_diff)
                rel_error = torch.where(non_nan, torch.abs(output_diff/labels), 0)

                mean_abs_error = torch.sum(abs_error, dim = -2) / torch.sum(non_nan, dim = -2)
                mean_rel_error = torch.sum(rel_error, dim = -2) / torch.sum(non_nan, dim = -2)
                
                
                val_SS_res +=  torch.sum(output_diff**2, dim = -2)
            
       
                rup_pred = torch.round(outputs[:,-1])
                acc = torch.sum(rup_pred == labels[:,-1])/len(rup_pred)
               
             
                # Add to list
                abs_error_list.append(mean_abs_error)
                rel_error_list.append(mean_rel_error)
                accuracy_list.append(acc.item())
            
            
            losses = np.array(losses)
            R2 = (1 - val_SS_res/self.val_SS_tot).to('cpu').numpy() 
            abs_error = torch.mean(torch.stack(abs_error_list), dim = -2).to('cpu').numpy()
            rel_error = torch.mean(torch.stack(rel_error_list), dim = -2).to('cpu').numpy()
            accuracy = np.mean(accuracy_list)
            
            # Put last to nan corresponding to is_ruptured
            abs_error[-1] = np.nan
            rel_error[-1] = np.nan
            R2[-1] = np.nan
            
            
            return np.mean(losses, axis = 0), R2, abs_error, rel_error, accuracy



    def forward_pass(self, data):
        # --- Evaluate --- #    
        image, vals = get_inputs(data, self.device)
        labels = get_labels(data, self.model.keys, self.device)
        outputs = self.model((image, vals))
        loss, Ff_loss, other_loss, rup_loss = self.criterion(outputs, labels)
        return loss, Ff_loss, other_loss, rup_loss, outputs, labels 
    
                
    def save_history(self, name):
        """ Save training history, best scores and model weights """
       
        # --- Save --- #
        save_training_history(name, self.history, self.get_info())
        save_best_model_scores(name, self.best, self.history, self.get_info())
        save_best_model(name, self.model, self.best['weights'])
    
    def plot_history(self, show = True, save = False):
        # Some quick plotting for insight
        plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
        start = 0
        if len(self.history['train_tot_loss']) > 10:
            start = 10
            
        plt.plot(self.history['train_tot_loss'][start:], '-o', markersize = 1, label = "Training")
        plt.plot(self.history['val_tot_loss'][start:], '-o',  markersize = 1, label = "Validation")
        plt.xscale('log')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize = 13)
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        if save is not False:
            plt.savefig(save, bbox_inches='tight')
        
        if show:
            plt.show()


    def get_info(self):
        s = '# --- Model settings --- #\n'
        s += f'# name = {self.model.name}\n'
        s += f'# image_shape = {self.model.image_shape}\n'
        s += f'# input_num = {self.model.input_num}\n'
        s += f'# conv_layers = {self.model.conv_layers}\n'
        s += f'# FC_layers = {self.model.FC_layers}\n'
        s += f'# out_features = {self.model.out_features}\n'
        s += f'# keys = {self.model.keys}\n'
        s += f'# mode = {self.model.mode}\n'
        s += f'# batchnorm = {self.model.batchnorm}\n'
        s += '# --- Model info --- #\n'
        s += f'# num_params = {self.model.get_num_params()}\n'
        s += f'# --- Criterion --- #\n'
        s += f'# alpha = {self.criterion.alpha}\n'
        s += f'# out_features = {self.criterion.out_features}\n'
        s += f'# criterion = {self.criterion.criterion}\n'
        s += f'# --- ML settings --- #\n'
        for key in self.ML_setting:
            s += f'# {key} = {self.ML_setting[key]}\n'
        s += f'# --- Data --- #\n'
        s += f'# data_root = {self.data_root}\n'
        
        return s
    
    def __str__(self):
        return self.get_info()

        
                


# --- Available data --- #
# stretch
# F_N
# scan_angle
# dt
# T
# drag_speed
# drag_length
# K
# stretch_speed_pct	
# relax_time
# pause_time1
# pause_time2
# rupture_stretch
# is_ruptured	
# Ff_max
# Ff_mean
# Ff_mean_std
# contact
# contact_std



if __name__=='__main__':
    # root = '../Data/ML_data/' # Relative (local)
    root = '/home/users/mikkelme/ML_data/' # Absolute path (cluster)
    data_root = [root+'baseline', root+'popup', root+'honeycomb', root+'RW']
    ML_setting = get_ML_setting()
    
    
    # Reference: Ff, rup_stretch, is_ruptured
    # alpha = [[1/2], [], [1/4, 1/4]]
    # criterion_out_features = [['R'], [], ['R', 'C']]
    # keys = ['Ff_mean', 'rupture_stretch', 'is_ruptured']
    # model_out_features = [item for sublist in criterion_out_features for item in sublist]        
    
    # + Contact
    # alpha = [[1/2, 1/6], [], [1/6, 1/6]]
    # criterion_out_features = [['R', 'R'], [], ['R', 'C']]
    # keys = ['Ff_mean', 'contact', 'rupture_stretch', 'is_ruptured']
    # model_out_features = [item for sublist in criterion_out_features for item in sublist]        
    
    # + Porosity
    # alpha = [[1/2], [1/6], [1/6, 1/6]]
    # criterion_out_features = [['R'], ['R'], ['R', 'C']]
    # keys = ['Ff_mean', 'porosity', 'rupture_stretch', 'is_ruptured']
    # model_out_features = [item for sublist in criterion_out_features for item in sublist]        
    
    # + Ff max
    # alpha = [[1/2, 1/6], [], [1/6, 1/6]]
    # criterion_out_features = [['R', 'R'], [], ['R', 'C']]
    # keys = ['Ff_mean', 'Ff_max', 'rupture_stretch', 'is_ruptured']
    # model_out_features = [item for sublist in criterion_out_features for item in sublist]        
    
    # Contact + Porosity + Ff max
    alpha = [[1/2, 1/10, 1/10], [1/10], [1/10, 1/10]]
    criterion_out_features = [['R', 'R', 'R'], ['R'], ['R', 'C']]
    keys = ['Ff_mean', 'Ff_max', 'contact', 'porosity', 'rupture_stretch', 'is_ruptured']
    model_out_features = [item for sublist in criterion_out_features for item in sublist]        
    
    # Training
    model = VGGNet( mode = 0, 
                    input_num = 2, 
                    conv_layers = [(1, 32), (1, 64), (1, 128), (1, 256), (1, 512), (1,1024)], 
                    FC_layers = [(1, 1024), (1,512), (1,256), (1, 128), (1, 64), (1,32)],
                    out_features = model_out_features,
                    keys = keys)
    


    criterion = Loss(alpha = alpha, out_features = criterion_out_features)
    
    ML_setting['max_epochs'] = 1000
    
    coach = Trainer(model, data_root, criterion, **ML_setting)
    coach.learn(max_epochs = None, max_file_num = None)
    coach.save_history('training/test_cyclic_S32D12')
    coach.plot_history(show = True, save = 'training/test_cyclic_S32D12/loss.pdf')

    # coach.plot_history()
    # coach.get_info()
    
    
    


    
    
    # coach = Trainer(model, data_root, loss, **ML_setting)
    # coach = Trainer(model, data_root, loss, use_gpu = True)
    
    
