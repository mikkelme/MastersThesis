from ML_utils import *
seed_everything(2023) # TODO: Keep here? Or put elsewhere?



class KirigamiDataset(Dataset):
    """ Imports the Kirigami Dataset and generates dataloaders. """
    
    def __init__(self, data_root, trvaltest, transform = None, maxfilenum = None, train_val_split = 0.2):
        """_summary_

        Args:
            data_root (String / List of strings): Can be parent folder of multiple paths...
            trvaltest (_type_): _description_
            transform (_type_, optional): _description_. Defaults to None.
            maxfilenum (_type_, optional): _description_. Defaults to None.
        """        
        
        
        self.data_root = data_root
        self.data_dir = []
        random_seed = 0 # Independent seed for tweeking train-val-split
       
        # --- Collect data dirs --- #
        indexes = [] # Store index from data names
        if isinstance(data_root, str): # Single path [string]
            indexes = self.collect_dirs(data_root, indexes)
            
        elif  hasattr(data_root, '__len__'): # Multiple paths [list of strings]
            for path in data_root:
                indexes = self.collect_dirs(path, indexes)
                
        else:
            print(f"Data root: {data_root}, not understood")
            exit()
        
    
        # --- Organize --- #
        # List to array
        indexes = np.array(indexes)     
        self.data_dir = np.array(self.data_dir)     
        
        # Sort by index
        sort_idx = np.argsort(indexes)  
        self.data_dir = self.data_dir[sort_idx]
        
    
        # (Optionally): Reduce number of images into RN sub sample
        if maxfilenum: 
            self.data_dir = np.random.choice(self.data_dir, maxfilenum, replace = False)
            # self.data_dir = self.data_dir[:maxfilenum]
            
        # Make train-validation-split
        train, val = train_test_split(np.arange(len(self.data_dir)), test_size = train_val_split, random_state = random_seed)

        # Select data corresponding to either training or validation
        if trvaltest == 'train':   # Train mode
            self.data_dir = self.data_dir[train]
        elif trvaltest == 'val':   # Validation mode
            self.data_dir = self.data_dir[val]
        else:
            print('Please Provide valid data mode: \"train\" or \"val\"')
            exit()
            
            
    def collect_dirs(self, path, indexes):
        """ Go through dirs and append """
        for dir in os.listdir(path):
            try:
                dir_idx = int(dir.split('_')[-1])
            except ValueError: # TODO: Remind me when this happens validly?
                continue
            indexes.append(dir_idx)
            self.data_dir.append(os.path.join(path, dir))
        return indexes
            
        
        
    def __len__(self):
        return len(self.data_dir)
    
    def __getitem__(self, idx):
        """ Get data sample """
        sample = {} # Sample dictionary 
        
        # Cut configuration
        sample['config'] = torch.from_numpy(np.load(os.path.join(self.data_dir[idx], 'config.npy')).astype(np.float32))
        
        # Get numerics
        with open(os.path.join(self.data_dir[idx], 'val.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile)
            for key, val in reader:
                sample[key] = val
        
        # Add directory (filename)
        sample['dir'] = self.data_dir[idx]
        return sample
        

def get_data(data_root, ML_setting, maxfilenum = None):
    """Get datasets and dataloaders from Kirigami dataset. """

    # Datasets
    datasets={}
    datasets['train'] = KirigamiDataset(data_root, trvaltest = 'train', maxfilenum = maxfilenum)
    datasets['val']   = KirigamiDataset(data_root, trvaltest = 'val',   maxfilenum = maxfilenum)

    # Dataloaders
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size = ML_setting['batchsize_train'], shuffle=True,  num_workers = 0) # when to have num_workers > 0 ?
    dataloaders['val']   = torch.utils.data.DataLoader(datasets['val'],   batch_size = ML_setting['batchsize_val'],   shuffle=False, num_workers = 0) # only for GPU perhaps?
    
    return datasets, dataloaders
    
def get_ML_setting(use_gpu = False):
    """ Fetch config dictionary with hardcoded settings  """
    
    ML_setting = {}
    ML_setting['use_gpu'] = use_gpu
    ML_setting['lr'] = 0.005                # Learning rate
    ML_setting['batchsize_train'] = 16   
    ML_setting['batchsize_val'] = 64
    ML_setting['maxnumepochs'] = 35
    ML_setting['scheduler_stepsize'] = 10
    ML_setting['scheduler_factor'] = 0.3

    return ML_setting


   
if __name__ == "__main__":
    # data_root = '../data_pipeline/tmp_data'
    # data_root = '../Data/ML/honeycomb'
    # data_root = '../Data/ML/popup'
    data_root = ['../Data/ML/honeycomb', '../Data/ML/popup'] 
    ML_setting = get_ML_setting()
    
    
    KirigamiDataset(data_root, trvaltest = 'train')
    
    # datasets, dataloaders = get_data(data_root, ML_setting, maxfilenum = None)
    # datasets, dataloaders = get_data(data_root, ML_setting, maxfilenum = None)
    # trainloader = dataloaders['train']
    
    
    
    

    
    # test_loader = KirigamiDataset(data_root, 'train')
    # for i in range(len(test_loader)):
    #     print(test_loader[i]['stretch'])
    
    # list of vowels
    # phones = ['apple', 'samsung', 'oneplus']
    # phones_iter = iter(phones)

    
    # print("go")
    # loader_iter = iter(trainloader)
    # for i in range(10):
    #     out = next(loader_iter)
    #     print(out)
    # print("go")
    # loader_iter = iter(trainloader)
    # for i in range(10):
    #     out = next(loader_iter)
    #     print(out)
        
        
        
    # num_batches = len(trainloader)
    # print(num_batches)
    # for batch_idx, data in enumerate(trainloader):
    #     print(batch_idx)
    # print('next')
    # for batch_idx, data in enumerate(trainloader):
    #     print(batch_idx)
