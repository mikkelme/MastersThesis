from module_import import *
from ML_utils import *
seed_everything(2023)



class KirigamiDataset(Dataset):
    """ Imports the Kirigami Dataset and generates dataloaders. """
    
    def __init__(self, data_root, trvaltest, transform = None, maxfilenum = None):
        self.data_root = data_root
        self.data_dir = []
        random_seed = 0
       
        # Collect data dirs
        indexes = []
        for dir in os.listdir(data_root):
            try:
                dir_idx = int(dir.split('_')[-1])
            except ValueError:
                continue
            indexes.append(dir_idx)
            self.data_dir.append(os.path.join(data_root, dir))
       
        # List to array
        indexes = np.array(indexes)     
        self.data_dir = np.array(self.data_dir)     
        
        # Sort by index
        sort_idx = np.argsort(indexes)
        self.data_dir = self.data_dir[sort_idx]
        
        # Reduce number of images | Make random version of this? XXX 
        if maxfilenum: # reduce number of images
            self.data_dir = self.data_dir[:maxfilenum]
      
        train, val = train_test_split(np.arange(len(self.data_dir)), test_size = 0.20, random_state = random_seed)

        if trvaltest == 'train':   # Train mode
            self.data_dir = self.data_dir[train]
        elif trvaltest == 'val':   # Validation mode
            self.data_dir = self.data_dir[val]
        else:
            print('Please Provide valid data mode: 0 = \"train\" or 1 = \"val\"')
            exit()
            
            
    def __len__(self):
        return len(self.data_dir)
    
    def __getitem__(self, idx):
        config = torch.from_numpy(np.load(os.path.join(self.data_dir[idx], 'config.npy')).astype(np.int32))
        
        sample = {}
        with open(os.path.join(self.data_dir[idx], 'val.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile)
            for key, val in reader:
                sample[key] = val
        
        sample['config'] = config
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
    ML_setting['batchsize_train'] = 1  # 16    
    ML_setting['batchsize_val'] = 64
    ML_setting['maxnumepochs'] = 35
    ML_setting['scheduler_stepsize'] = 10
    ML_setting['scheduler_factor'] = 0.3

    return ML_setting


   
if __name__ == "__main__":
    data_root = '../data_pipeline/tmp_data'
    ML_setting = get_ML_setting()
    
    datasets, dataloaders = get_data(data_root, ML_setting, maxfilenum = None)
    trainloader = dataloaders['train']
    
    
    
    

    
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
