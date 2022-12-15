import numpy as np
import os

import sys
sys.path.append('../') # parent folder: MastersThesis
from config_builder.build_config import *
from simulation_manager.multi_runner import *


class data_generator:
    def __init__(self, mat):
        shape_error = f"SHAPE ERROR: Got matrix of shape {np.shape(mat)}, y-axis must be multiple of 2 and both nonzero integer."
        assert mat.shape[0]%1 == 0 and mat.shape[1]%1 == 0 and mat.shape[1]%2 == 0, shape_error
        assert mat.shape[0] > 0 and mat.shape[1] > 0, shape_error
        
        self.mat = mat
        self.Cdis = 1.461 # carbon-carbon distance [Å]         

        self.shape = np.shape(mat)            
        # self.set_sheet_size()
        
        
        # self.dir = "egil:CONFIGS"
        # self.dir = "./CONFIGS"
        
        
    def set_sheet_size(self):
        """ Set sheet size according to configuration matrix shape """
        xlen = self.shape[0]
        ylen = self.shape[1]//2
        a = 3*self.Cdis/np.sqrt(3)
        
        self.Lx = a/6*np.sqrt(3) + (xlen-1) * a/2*np.sqrt(3)
        self.Ly = a/2 + (ylen-1) * a
        
    
    def get_substrate_size(self, stretch_pct, margins = (10, 10)):
        """ Get substrate size considered sretch and x,y-margines """
        Lx = self.Lx + 2*margins[0]
        Ly = self.Ly * (1 + stretch_pct) + 2*margins[1]
        return Lx, Ly
        
        
    def run(self):
        config_ext = 'tmp'
        config_path = '.'
        
        
        # Intialize simulation runner
        proc = Simulation_runner()
        
        # Build sheet 
        builder = config_builder(self.mat)
        builder.add_pullblocks()
        builder.save("sheet", ext = config_ext, path = config_path)
        proc.add_variables(config_data = f'sheet_{config_ext}' )
        
        # Directories 
        header = "egil:CONFIGS/test1"
        dir = os.path.join(header, "conf1")
        proc.config_path = config_path
        
        
        # Multi run settings 
        num_stretch_files = 5
        F_N = np.sort(np.random.uniform(0.1, 1, 5))*1e-9
        # F_N = np.linspace(0.1e-9, 1e-9, 3)
        
        proc.add_variables(num_stretch_files = num_stretch_files, 
                           RNSEED = '$RANDOM',
                           run_rupture_test = 1)
        

        # Start multi run
        proc.multi_run(header, dir, F_N, num_procs = 16, jobname = "CONF")
       
        # Remove config files after transfering
        os.remove(os.path.join(config_path,f'sheet_{config_ext}.txt'))
        os.remove(os.path.join(config_path,f'sheet_{config_ext}_info.in'))

    

      


    

def read_configurations(folder):
    configs = []
    # print(f"Reading configurations | dir: {folder}")
    for file in os.listdir(folder):
        try:
            mat = np.load(os.path.join(folder,file))
            configs.append(mat)
            # print("√ |", file)
            
        except ValueError:
            pass
            # print("X |", file)
    return configs












if __name__ == "__main__":
    configs = read_configurations("../graphene_sheet/test_data")
    conf = data_generator(configs[0])
    conf.run()
    
    
    # print(conf.Lx, conf.Ly)
    # conf.get_substrate_size(0.20)
    
    
 
    
    
    # gen = data_generator()
    # gen.read_configurations("../graphene_sheet/test_data" )
    # gen.get_sheet_size(gen.configs[0])