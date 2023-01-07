import numpy as np
import os

import sys
sys.path.append('../') # parent folder: MastersThesis
from config_builder.build_config import *
from simulation_manager.multi_runner import *
from analysis.analysis_utils import get_files_in_folder

class data_generator:
    def __init__(self, filename, header =  'egil:CONFIGS/sizes', simname = 'conf', config_ext = None):#, config_ext):
        
        try:
            self.mat = np.load(filename)
            assert self.mat.shape[0]%1 == 0 and self.mat.shape[1]%1 == 0 and self.mat.shape[1]%2 == 0
            assert self.mat.shape[0] > 0 and self.mat.shape[1] > 0
            
        
        except ValueError:
            print(f"Could not load file: {filename}")
            return # XXX
        except FileNotFoundError:
            print(f"File not found: {filename}")
            return # XXX
        except AssertionError:
            print(f"SHAPE ERROR: Got matrix of shape {np.shape(self.mat)}, y-axis must be multiple of 2 and both nonzero integer.")
            return # XXX
        
        if config_ext is None:
            self.config_ext = filename.split('/')[-1].split('.')[0]
        else:
            self.config_ext = config_ext
        
        
        self.Cdis = 1.461 # carbon-carbon distance [Å]         
        self.shape = np.shape(self.mat)            
        
        # Path
        self.npy_file = filename # Remember array filename
        self.header =  header
        self.dir = os.path.join(self.header, simname)
        
        
        
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
        
        # # Build sheet 
        builder = config_builder(self.mat)
        builder.add_pullblocks()
        png_file                            = builder.save_view('sheet', path = config_path)
        lammps_file_txt, lammps_file_info   = builder.save_lammps("sheet", ext = self.config_ext, path = config_path)
        config_data = f'sheet_{self.config_ext}'
        proc.add_variables(config_data = config_data)
        
        # Directories 
        proc.config_path = config_path
        
        # Multi run settings 
        num_stretch_files = 10
        F_N = np.sort(np.random.uniform(0.1, 10, 10))*1e-9
        # F_N = np.array([1, 10])*1e-9
        # F_N = np.linspace(0.1e-9, 1e-9, 3)
        
        proc.add_variables(num_stretch_files = num_stretch_files, 
                           RNSEED = '$RANDOM',
                           run_rupture_test = 1,
                           stretch_max_pct = 0.7,
                           root = '.',
                           dump_freq = 100000)
        
        
        # Start multi run
        root_path = proc.multi_run(self.header, self.dir, F_N, num_procs = 16, jobname = "cut_sizes")
        
        # Transfer config npy- and png-file 
        proc.move_files_to_dest([self.npy_file, png_file], root_path)
       
        # Remove generated files locally
        os.remove(png_file)
        os.remove(lammps_file_txt)
        os.remove(lammps_file_info)
        


def run_files(filenames, header):
    for file in filenames:
        gen = data_generator(file, header)
        gen.run()
        


if __name__ == "__main__":
    run_files(get_files_in_folder('../config_builder/nocut_sizes/', exclude = 'DS_Store'), header =  'egil:CONFIGS/nocut_sizes')
    
    # gen = data_generator('../config_builder/cut_nocut/cut1.npy')
    # gen = data_generator('../config_builder/sizes/cut_42x24.npy')
    # gen.run()
   
    