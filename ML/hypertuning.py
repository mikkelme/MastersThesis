
from train_network import *



class Architectures:
    def __init__(self, mode = 0, batchnorm = True):
        """
            Architectures suggested in article on 
            graphene/h-BN interface https://doi.org/10.1063/5.0131576
        """
        # Count number of methods starting with 'A' corresponding to architectures
        self.a = []
        for d in (d for d in dir(self) if d[0] == 'A'):
            self.a.append(eval('self.'+d))
            
        # --- Shared settings --- #
        # Model
        self.mode = mode
        self.batchnorm = batchnorm
        
        # Common data settings
        self.common_settings()
        
    def common_settings(self):
        pass
       
    
    def __str__(self):
        if len(self) == 0:
            return 'No architecture methods implemented.'
        
        s = f'Architecture(s) implemented = {len(self)}:\n'
        for i, (model, criterion) in enumerate(self):
            s += f'{i} | {model.name} (#params = {model.get_num_params()})\n'
        return s
        
    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        return self.a[idx]()
        
        
  
def train_architectures(A_instance, data_root, ML_setting, save_folder):
    for i, (model, criterion) in enumerate(A_instance):
        coach = Trainer(model, data_root, criterion, **ML_setting)
        coach.learn()
        coach.save_history(os.path.join(save_folder, model.name))
        coach.plot_history(show = False, save = os.path.join(save_folder, f'{model.name}_loss.pdf'))
      
      
  

if __name__ == '__main__':
    root = '../Data/ML_data/'
    data_root = [root+'baseline', root+'popup', root+'honeycomb']


    # model, criterion = get_A(1)

    A = Architectures()
    exit(A)
    
    # TODO: Try to run this one 
    model, criterion = A[7]
    coach = Trainer(model, data_root, criterion)
    coach.learn(max_epochs = 300, max_file_num = None)
    coach.save_history('training/test')
    coach.plot_history()
