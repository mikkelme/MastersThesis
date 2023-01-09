import torch
# import torch.nn as nn
# import torch.optim as optim
from torch.utils.data import Dataset
# from torchvision import models, transforms

import os
# import PIL.Image
# import pandas as pd

# from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# from sklearn.metrics import average_precision_score
import numpy as np
import matplotlib.pyplot as plt


import csv

class KirigamiDataset(Dataset):
    """ Imports the Kirigami Dataset and
    generates dataloaders. """
    
    def __init__(self, data_root, trvaltest, transform=None, maxfilenum = None):
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
        
        # Divide into train / validation TODO: What about test?
        train, val = train_test_split(np.arange(len(self.data_dir)), test_size = 0.20, 
                                                                         random_state = random_seed)

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
        config = torch.from_numpy(np.load(os.path.join(self.data_dir[idx], 'config.npy')))
        
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
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size = ML_setting['batchsize_train'], shuffle=True,  num_workers = 1)
    dataloaders['val']   = torch.utils.data.DataLoader(datasets['val'],   batch_size = ML_setting['batchsize_val'],   shuffle=False, num_workers = 1)
    
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
    data_root = 'tmp_data'
    ML_setting = get_ML_setting(data_root)
    
    sample = KirigamiDataset(data_root, 'train')[0]
    
    # print(sample['config'])
    
    datasets, dataloaders = get_data(data_root, ML_setting, maxfilenum = 100)
    trainloader = dataloaders['train']
    # out = next(iter(trainloader))
    
    num_batches = len(trainloader)
    print(num_batches)
    for batch_idx, data in enumerate(trainloader):
        print(batch_idx)
        exit()
