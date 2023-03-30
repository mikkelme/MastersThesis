from train_network import *
from time import perf_counter

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
        
        
  
def train_architectures(A_instance, data_root, ML_setting, save_folder):
    timer_file = os.path.join(save_folder, 'timings.txt')
    for i, (model, criterion) in enumerate(A_instance):
        if i < 27: continue
        num_params = model.get_num_params()
        print(f'{i} | {model.name} (#params = {num_params:1.2e})')
        timer_start = perf_counter() 
        
        try:
            coach = Trainer(model, data_root, criterion, **ML_setting)
            coach.learn()
            coach.save_history(os.path.join(save_folder, model.name))
            coach.plot_history(show = False, save = os.path.join(save_folder, model.name, 'loss.pdf'))
            plt.figure().close('all') 
        except: # weights exploted inside or something
            print(f'Crashed at architecture {i}')
            pass
        
        timer_stop = perf_counter()
        elapsed_time = timer_stop - timer_start
        h = int(elapsed_time // 3600)
        m = int((elapsed_time % 3600) // 60)
        s = int(elapsed_time % 60)
    

        if i == 0: # Create file
            outfile = open(timer_file, 'w')
            outfile.write('# Architecture | time [h:m:s]\n')
            outfile.write(f'{i} | {model.name} (#params = {num_params:1.2e}) {h:02d}:{m:02d}:{s:02d}\n')
        else: # Append to file
            outfile = open(timer_file, 'a')
            outfile.write(f'{i} | {model.name} (#params = {num_params:1.2e}) {h:02d}:{m:02d}:{s:02d}\n')
        outfile.close()
            


if __name__ == '__main__':
    root = '../Data/ML_data/'
    data_root = [root+'baseline', root+'popup', root+'honeycomb']




    # A = Architectures()
    
    # model, criterion = A[6]
    # coach = Trainer(model, data_root, criterion)
    # coach.learn(max_epochs = 2, max_file_num = None)
    # coach.save_history('training')
    # coach.plot_history()
