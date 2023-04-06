import sys
sys.path.append('../') # parent folder: MastersThesis

if 'MastersThesis' in sys.path[0]: # Local 
    from ML.train_network import *
else: # Cluster
    from train_network import *
    
    
from time import perf_counter
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import FastaiLRFinder
from matplotlib.colors import LogNorm
from scipy.signal import argrelextrema







class Architectures:
    def __init__(self, mode = 0, batchnorm = True):
        # Model
        self.mode = mode
        self.batchnorm = batchnorm
        
        # Architecture list
        self.A = [] # append (model, criterion)
        self.initialize()
            
        
    def initialize(self):
        pass
       
    
    def __str__(self):
        if len(self) == 0:
            return 'No architecture methods implemented.'
        
        s = f'Architecture(s) implemented = {len(self)}:\n'
        for i, (model, criterion) in enumerate(self):
            # num_params = model.get_num_params()*1e-3 # in thousands
            # s += f'{i} | {model.name} (#params = {num_params:5.3f}k)\n'
            num_params = model.get_num_params()
            s += f'{i} | {model.name} (#params = {num_params:1.2e})\n'
        return s
        
    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        return self.A[idx]
        
        
        
class best_model(Architectures):    
    def initialize(self):
        # Data outputs
        alpha = [[1/2, 1/10, 1/10], [1/10], [1/10, 1/10]]
        criterion_out_features = [['R', 'R', 'R'], ['R'], ['R', 'C']]
        keys = ['Ff_mean', 'Ff_max', 'contact', 'porosity', 'rupture_stretch', 'is_ruptured']
        model_out_features = [item for sublist in criterion_out_features for item in sublist]   
        criterion = Loss(alpha = alpha, out_features = criterion_out_features)
    
    
        s = 32; d = 12    
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
        

  
def train_architectures(A_instance, data_root, ML_setting, save_folder, LR_range = None):
    timer_file = os.path.join(save_folder, 'timings.txt')
    
    if LR_range:
        name = []; num_params = []; lr = []
        infile = open(LR_range, 'r')
        for line in infile:
            if line[0] == '#': continue
            
            name.append(line.split('|')[-1].split('(')[0].strip(' '))
            num_params.append(float(line.split('params = ')[1].split(')')[0]))
            lr.append(float(line.split('lr = ')[-1]))
        assert len(lr) == len(A_instance), f'Number of provided lr ({len(lr)}) is not matching the number of architectures ({len(A_instance)})'
        
        
        
    for i, (model, criterion) in enumerate(A_instance):  
        if LR_range:
            assert model.name == name[i], f'model name {model.name} does not match model name {name[i]} corresponding to LR_range test.'
            ML_setting['lr'] = lr[i]

        # if i < 41: continue
        crashed = False
        num_params = model.get_num_params()
        print(f'{i} | {model.name} (#params = {num_params:1.2e}), lr = {ML_setting["lr"]}')
        timer_start = perf_counter() 
        
        try:
            coach = Trainer(model, data_root, criterion, **ML_setting)
            coach.learn()
            coach.save_history(os.path.join(save_folder, model.name))
            coach.plot_history(show = False, save = os.path.join(save_folder, model.name, 'loss.pdf'))
            # plt.figure().close('all') # XXX Doesn't work yet
        except: # weights exploted inside or something
            print(f'Crashed at architecture {i}')
            crashed = True
        
        timer_stop = perf_counter()
        elapsed_time = timer_stop - timer_start
        h = int(elapsed_time // 3600)
        m = int((elapsed_time % 3600) // 60)
        s = int(elapsed_time % 60)
    

        if i == 0: # Create file
            outfile = open(timer_file, 'w')
            outfile.write('# Architecture | time [h:m:s]\n')
            outfile.write(f'{i} | {model.name} (#params = {num_params:1.2e}) {h:02d}:{m:02d}:{s:02d}')
        else: # Append to file
            outfile = open(timer_file, 'a')
            outfile.write(f'{i} | {model.name} (#params = {num_params:1.2e}) {h:02d}:{m:02d}:{s:02d}')
            
        if crashed:
            outfile.write(' (crashed)')
        outfile.write('\n')
        outfile.close()
            
            
            
def momentum_weight_search(model, criterion, data_root, ML_setting, max_lr, momentum, weight_decay, save_folder):
    timer_file = os.path.join(save_folder, 'timings.txt')
    
    # max_lr = [0.19098169151133398, 0.07099694630794687, 0.048523309416493646, 0.0078083987538332695]
    # momentum = [0.99, 0.97, 0.95, 0.9]
    # weight_decay = [1e-4, 1e-5, 1e-6, 0]    

    # Get settings
    cyclic_lr_div_factor, _, final_div_factor = ML_setting['cyclic_lr']
    cyclic_momentum_base, _ = ML_setting['cyclic_momentum']
    
    
    # Get model state for resetting
    state = model.state_dict()
    
    for i in range(len(momentum)):
        ML_setting['cyclic_lr'] = [cyclic_lr_div_factor, max_lr[i], final_div_factor]
        ML_setting['cyclic_momentum'] = [cyclic_momentum_base, momentum[i]]
        
        for j in range(len(weight_decay)):
            ML_setting['weight_decay'] = weight_decay[j]
            crashed = False
        
            if i == 0 and j == 0:
                continue
        
            info_string = f'cyclic lr = ({ML_setting["cyclic_lr"][0]:.1f}, {ML_setting["cyclic_lr"][1]:.2e}, {ML_setting["cyclic_lr"][2]:.1e}), cyclic momentum = ({ML_setting["cyclic_momentum"][0]:.1f}, {ML_setting["cyclic_momentum"][1]:.2f}), weight decay = {ML_setting["weight_decay"]}'
            save_name = f'm{i}w{j}'
            
            print(f'{i} | {save_name} | {info_string}')
            timer_start = perf_counter() 
            
            model.load_state_dict(state) # Reset model before training again
            try:
                coach = Trainer(model, data_root, criterion, **ML_setting)
                coach.learn()
                coach.save_history(os.path.join(save_folder, save_name))
                coach.plot_history(show = False, save = os.path.join(save_folder, save_name, 'loss.pdf'))
                # plt.figure().close('all') # XXX Doesn't work yet
                
            except: # weights exploted inside or something
                print(f'Crashed at architecture {i}')
                crashed = True
            
            timer_stop = perf_counter()
            elapsed_time = timer_stop - timer_start
            h = int(elapsed_time // 3600)
            m = int((elapsed_time % 3600) // 60)
            s = int(elapsed_time % 60)
            s_timing = f'{h:02d}:{m:02d}:{s:02d}'

            if i == 0 and j == 0: # Create file
                outfile = open(timer_file, 'w')
                outfile.write('# Architecture | time [h:m:s]\n')
                outfile.write(f'{save_name} | {info_string} | {s_timing}')
            else: # Append to file
                outfile = open(timer_file, 'a')
                outfile.write(f'{save_name} | {info_string} | {s_timing}')
                
            if crashed:
                outfile.write(' (crashed)')
            outfile.write('\n')
            outfile.close()
                    


class Find_optimal_LR:
    def __init__(self, model, optimizer, criterion, data_root, ML_setting):
        self.model = model
        self.optimizer = optimizer 
        self.criterion = lambda x, y: criterion(x, y)[0] # Output only relevant loss
        # self.criterion = criterion
        self.data_root = data_root 
        self.ML_setting = ML_setting
        self.device = get_device(ML_setting)
        self.datasets, self.dataloaders = get_data(data_root, ML_setting)
        
        

    def prepare_batch_fn(self, batch, device, non_blocking):
        image, vals = get_inputs(batch, device)
        x = (image, vals)
        y = get_labels(batch, self.model.keys, device)
        return (x, y)
            
    
    def find_optimal(self, end_lr):
        train_loader = self.dataloaders['train']
        self.lr_finder = FastaiLRFinder()
        trainer = create_supervised_trainer(self.model, self.optimizer, self.criterion, device=self.device, prepare_batch=self.prepare_batch_fn)
        
        # epoch_length = trainer.state.epoch_length
        # max_epochs = trainer.state.max_epochs
        # print(epoch_length, max_epochs)
        # exit()
        
        # To restore the model's and optimizer's states after running the LR Finder
        to_save = {"model": self.model, "optimizer": self.optimizer}
        
        with self.lr_finder.attach(trainer, to_save, end_lr = end_lr) as trainer_with_lr_finder:
            trainer_with_lr_finder.run(train_loader)
        
        # results = lr_finder.get_results()
        # sugestion = lr_finder.lr_suggestion()
        # return results, sugestion
    
        # lr_finder.plot()
        
        # print("Suggested LR", self.lr_finder.lr_suggestion())

    
def LR_range_test(A_instance, data_root, ML_setting, save_folder):
    start_lr = 1e-7
    end_lr = 10.0

    # optimizer = optim.Adam(model.parameters(), lr = 1e-6)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-06)
    # optimizer = optim.SGD(model.parameters(), lr = 1e-06, momentum = 0.9)    
    filename = os.path.join(save_folder, 'lr.txt')
    for i, (model, criterion) in enumerate(A_instance):        
        num_params = model.get_num_params()
        
        optimizer = optim.Adam(model.parameters(), lr = start_lr)
        foLR = Find_optimal_LR(model, optimizer, criterion, data_root, ML_setting)
        try:
            foLR.find_optimal(end_lr)
            suggestion = foLR.lr_finder.lr_suggestion()
        except RuntimeError:
            print('Runtime Error')
            suggestion = np.nan
            
            
        if i == 0: # Create file
            try:
                outfile = open(filename, 'w')
            except FileNotFoundError:
                os.makedirs(save_folder)
                outfile = open(filename, 'w')
            outfile.write('# Architecture | suggested lr\n')
            outfile.write(f'{i} | {model.name} (#params = {num_params:1.2e}) suggested lr = {suggestion}\n')
        else: # Append to file
            outfile = open(filename, 'a')
            outfile.write(f'{i} | {model.name} (#params = {num_params:1.2e}) suggested lr = {suggestion}\n')
        outfile.close()
            
        print(f'{i} | {model.name} (#params = {num_params:1.2e}), suggestion = {suggestion}')
    


def plot_LR_range_test(path): 
    infile = open(path, 'r')
    
    D = []
    S = []
    num_params = []
    lr = []
    for line in infile:
        if line[0] == '#': continue
        name = line.split('|')[-1].split('(')[0].strip(' ')
        S.append(int(name.split('D')[0].strip('S')))
        D.append(int(name.split('D')[1].strip('D')))
        num_params.append(float(line.split('params = ')[1].split(')')[0]))
        lr.append(float(line.split('lr = ')[-1]))
        if name == 'S64D4': break

    
    # --- Organize into matrix (D, S, lr) --- #
    # Get unique axis
    D_axis =  np.sort(list(set(D)))
    S_axis =  np.sort(list(set(S)))
    shape = (len(D_axis), len(S_axis))
    
    # Get 1D -> 2D mapping
    D_mat, S_mat = np.meshgrid(D_axis, S_axis)
    map = np.full(np.shape(D_mat), -1)
    for i in range(D_mat.shape[0]):
        for j in range(D_mat.shape[1]):
            D_hit = D_mat[i, j] == D
            S_hit = S_mat[i, j] == S
            full_hit = np.logical_and(D_hit, S_hit)
            if np.sum(full_hit) == 1:
                map[i,j] = int(np.argmax(full_hit))
            elif np.sum(full_hit) > 1:
                exit('This should not happen')
                
    # Flip axis for increasing y-axis
    map = np.flip(map, axis = 0)
    S_axis = np.flip(S_axis) 
    
    # Perform mapping
    D = np.array(D + [np.nan])[map]
    S = np.array(S + [np.nan])[map]
    num_params = np.array(num_params + [np.nan])[map]
    lr = np.array(lr + [np.nan])[map]
    
    
    
    # --- Plotting --- #
    # Heatmap
    fig, ax = plt.subplots(num = unique_fignum(), figsize=(10, 6))
    vmin, vmax = None, None
    sns.heatmap(lr, xticklabels = D_axis, yticklabels = S_axis, cbar_kws={'label': 'Suggested lr'}, annot=True, fmt='.2e',  norm=LogNorm(), vmin=vmin, vmax=vmax, ax=ax)
    ax.set_xlabel('Depth', fontsize=14)
    ax.set_ylabel('Start num. channels', fontsize=14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)

    # Complexity
    sort = np.argsort(num_params.flatten())
    num_params = (num_params.flatten())[sort]
    lr = (lr.flatten())[sort]


    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(num_params, lr)
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Num. params', fontsize=14)
    plt.ylabel('Learning rate', fontsize=14)
    # plt.legend(fontsize = 13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    # plt.savefig('../article/figures/figure.pdf', bbox_inches='tight')
    plt.show()







if __name__ == '__main__':
    root = '../Data/ML_data/'
    data_root = [root+'baseline', root+'popup', root+'honeycomb', root+'RW']
    data_root = [root+'honeycomb']
    
    # plot_LR_range_test('staircase_lr/lr_staircase.txt')
    # plt.show()
    
        
        
    # ML_setting = {
    #     'use_gpu': False,
    #     'lr': 0.01, 
    #     'batchsize_train': 32,
    #     'batchsize_val': 64,
    #     'max_epochs': 5, #1000,
    #     'max_file_num': None,
    # }

    # alpha = [[1/2, 1/10, 1/10], [1/10], [1/10, 1/10]]
    # criterion_out_features = [['R', 'R', 'R'], ['R'], ['R', 'C']]
    # keys = ['Ff_mean', 'Ff_max', 'contact', 'porosity', 'rupture_stretch', 'is_ruptured']
    # model_out_features = [item for sublist in criterion_out_features for item in sublist]        
    
    # # Training
    # model = VGGNet( mode = 0, 
    #                 input_num = 2, 
    #                 conv_layers = [(1,8), (1,16), (1,32), (1,64)], 
    #                 FC_layers = [(1,64), (1,32), (1,16), (1,8)],
    #                 out_features = model_out_features,
    #                 keys = keys)
    
    # criterion = Loss(alpha = alpha, out_features = criterion_out_features)
    # optimizer = optim.Adam(model.parameters(), lr = 1e-6)
    # # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-06)
    # # optimizer = optim.SGD(model.parameters(), lr = 1e-06, momentum = 0.9)
    

    # foLR = Find_optimal_LR(model, optimizer, criterion, data_root, ML_setting)
    # foLR.find_optimal(end_lr = 1.0)
