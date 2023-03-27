
from train_network import *



# class Architectures:
#     def __init__(self, mode = 0, batchnorm = True):
#         """
#             Architectures suggested in article on 
#             graphene/h-BN interface https://doi.org/10.1063/5.0131576
#         """
#         # --- Shared settings --- #
#         # Model
#         self.mode = mode
#         self.batchnorm = batchnorm
        
#         # Common data settings
#         self.common_settings()
        
        
#         # Count number of methods starting with 'A' corresponding to architectures
#         self.a = []
#         for d in (d for d in dir(self) if d[0] == 'A'):
#             self.a.append(eval('self.'+d))
            
        
#     def common_settings(self):
#         pass
       
    
#     def __str__(self):
#         if len(self) == 0:
#             return 'No architecture methods implemented.'
        
#         s = f'Architecture(s) implemented = {len(self)}:\n'
#         for i, (model, criterion) in enumerate(self):
#             s += f'{i} | {model.name} (#params = {model.get_num_params()})\n'
#         return s
        
#     def __len__(self):
#         return len(self.a)

#     def __getitem__(self, idx):
#         return self.a[idx]()
        
        

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
            num_params = model.get_num_params()*1e-3 # in thousands
            s += f'{i} | {model.name} (#params = {num_params:5.3f}k)\n'
        return s
        
    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        return self.A[idx]
        
        
  
def train_architectures(A_instance, data_root, ML_setting, save_folder):
    for i, (model, criterion) in enumerate(A_instance):
        coach = Trainer(model, data_root, criterion, **ML_setting)
        coach.learn()
        coach.save_history(os.path.join(save_folder, model.name))
        coach.plot_history(show = False, save = os.path.join(save_folder, model.name, 'loss.pdf'))
      
      
  

if __name__ == '__main__':
    root = '../Data/ML_data/'
    data_root = [root+'baseline', root+'popup', root+'honeycomb']



    # A = Architectures()
    
    # model, criterion = A[6]
    # coach = Trainer(model, data_root, criterion)
    # coach.learn(max_epochs = 2, max_file_num = None)
    # coach.save_history('training')
    # coach.plot_history()
