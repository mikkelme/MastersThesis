import numpy as np
import os

import sys
sys.path.append('../') # parent folder: MastersThesis
from config_builder.build_config import *
from simulation_manager.multi_runner import *


class data_generator:
    def __init__(self, mat, config_ext):
        shape_error = f"SHAPE ERROR: Got matrix of shape {np.shape(mat)}, y-axis must be multiple of 2 and both nonzero integer."
        assert mat.shape[0]%1 == 0 and mat.shape[1]%1 == 0 and mat.shape[1]%2 == 0, shape_error
        assert mat.shape[0] > 0 and mat.shape[1] > 0, shape_error
        
        self.mat = mat
        
        # TODO: Consider reading array from file to ease the flow
        # for sending the saved array to the cluster as well
        
        
        self.Cdis = 1.461 # carbon-carbon distance [Å]         
        self.shape = np.shape(mat)            
        
        self.header =  "egil:CONFIGS/cut_nocut"
        self.dir = os.path.join(self.header, "conf")
        
        self.config_ext = config_ext
        
        
    def set_sheet_size(self):
        """ Set sheet size according to configuration matrix shape """
        xlen = self.shape[0]
        ylen = self.shape[1]//2
        a = 3*self.Cdis/np.sqrt(3)
        
        self.Lx = a/6*np.sqrt(3) + (xlen-1) * a/2*np.sqrt(3)
        self.Ly = a/2 + (ylen-1) * a
        
    
    def get_substrate_size(self, stretch_pct, margins = (10, 10)):
        """ Get substrate size considered sretch and x,y-margines """
        self.set_sheet_size()
        Lx = self.Lx + 2*margins[0]
        Ly = self.Ly * (1 + stretch_pct) + 2*margins[1]
        return Lx, Ly
        
        
    def run(self):
        config_path = '.'
        
        # Intialize simulation runner
        proc = Simulation_runner()
        
        # Build sheet 
        builder = config_builder(self.mat)
        builder.add_pullblocks()
        builder.save("sheet", ext = self.config_ext, path = config_path)
        config_data = f'sheet_{self.config_ext}'
        proc.add_variables(config_data = config_data)
        
        # Directories 
        proc.config_path = config_path
        
        
        # Multi run settings 
        num_stretch_files = 10
        F_N = np.sort(np.random.uniform(0.1, 10, 10))*1e-9
        # F_N = np.linspace(0.1e-9, 1e-9, 3)
        
        proc.add_variables(num_stretch_files = num_stretch_files, 
                           RNSEED = '$RANDOM',
                           run_rupture_test = 1,
                           stretch_max_pct = 0.7)
        

        # Start multi run
        proc.multi_run(self.header, self.dir, F_N, num_procs = 16, jobname = "Dgen2")
       
        # Remove config files after transfering
        os.remove(os.path.join(config_path,f'{config_data}.txt'))
        os.remove(os.path.join(config_path,f'{config_data}_info.in'))

    
class configuration_manager():
    def __init__(self):
        self.configs = []
        self.config_ext = []
    
    def add(self, path):
        try:
            mat = np.load(path)
        except ValueError:
            return False
        
        self.configs.append(mat)
        return True
            
    
    def read_folder(self, folder):
        print(f"Reading configurations | dir: {folder}")
        rejected = []
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            success = self.add(path)    
            if success:
                self.config_ext.append(file.split('.')[-2])
                print(f'\r√ |, {file}', end = "")
                
            else:
                rejected.append(file)
                print(f'\rX |, {file}', end = "")
                
        if len(rejected) == 0:
            print(f'\rRead folder succesfully.')
        else:
            string = ('\n').join([s for s in rejected])
            print(f'\rRejected:                       \n{string}', )
    
    def run_all(self):
        all_unique = (len(set(self.config_ext)) == len(self.config_ext))
        if all_unique:
            for (mat, ext) in zip(self.configs, self.config_ext):
                gen = data_generator(mat, ext)
                gen.run() # settings hardcoded/defined in data_generator for now XXX
        else:
            print("Error: Extension names are not unique all")
            exit(f'Found {len(self.config_ext) - len(set(self.config_ext))} repeated value(s).')

    def __str__(self):
        string = "------------------------\n"
        string += f'Num configs = {len(self.configs)}\n'
        string +=  ('\n').join([s for s in self.config_paths])
        string += "\n------------------------"
        return string





if __name__ == "__main__":
    configs = configuration_manager()
    configs.read_folder('../config_builder/cut_nocut/')
    # configs.run_all()
    
    
    