from dataloaders import *



class Data_fetch():
    """ Fetch data for analysing (not ML) """
    
    # Partial inheritance from KirigamiDataset 
    collect_dirs = KirigamiDataset.collect_dirs
    __len__ = KirigamiDataset.__len__
    
    def __init__(self, data_root):
        self.data_root = data_root
        self.data_dir = []
        self.sample = None

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
        
        
    def get_data(self, target, idx):
        self.data = {}
        for key in target:
            self.data[key] = []
        
        for i in idx:
            with open(os.path.join(self.data_dir[i], 'val.csv'), newline='') as csvfile:
                reader = csv.reader(csvfile)
                for key, val in reader:
                    if key in self.data:
                        self.data[key].append(val)
        
        for key in self.data:
            self.data[key] = np.array(self.data[key], dtype = float)
        
        return self.data
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return np.array([idx])
        else:
            return np.arange(len(self))[idx]
        
    

    def covariance_matrix(self):
        """ Use package from sklearn os somehting """
        # Pearsons correlation or?
        # TODO: Working here, make correlation or covariation matrix to 
        # illustrate correlations in the data.
        var = [self.data[keys] for keys in self.data]
        cov = np.cov(var)
        print(cov)
        # cov =  
    

# covariance = np.cov(, data2)
        

        

if __name__ == '__main__':
    data_root = ['../Data/ML_data/honeycomb', '../Data/ML_data/popup'] 
    data_root = '../Data/ML_data/honeycomb' 
    obj = Data_fetch(data_root)
    
    obj.get_data(['stretch', 'F_N'], obj[0:100])
    mat = obj.covariance_matrix()
    
    
    
    # obj.set_target(['stretch'])
    # sample = obj[0:2]
    # print(sample)
    