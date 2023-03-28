from train_network import *



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
        try:
            coach = Trainer(model, data_root, criterion, **ML_setting)
            coach.learn()
            coach.save_history(os.path.join(save_folder, model.name))
            coach.plot_history(show = False, save = os.path.join(save_folder, model.name, 'loss.pdf'))
        except: # weights exploted inside or something
            continue
      
      
  

if __name__ == '__main__':
    root = '../Data/ML_data/'
    data_root = [root+'baseline', root+'popup', root+'honeycomb']



    # A = Architectures()
    
    # model, criterion = A[6]
    # coach = Trainer(model, data_root, criterion)
    # coach.learn(max_epochs = 2, max_file_num = None)
    # coach.save_history('training')
    # coach.plot_history()
