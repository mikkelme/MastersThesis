import numpy as np
import os

import sys
sys.path.append('../') # parent folder: MastersThesis
from config_builder.build_config import *
from simulation_manager.multi_runner import *
from analysis.analysis_utils import get_files_in_folder

class Data_generator:
    def __init__(self, filename, header =  'egil:CONFIGS/honeycomb', simname = 'test', config_ext = None):
        
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
        self.config_path = '.' # Where to save new files temporary


    def __str__(self):
        s = 'class: Data_generator\n'
        s += f'filename: {self.npy_file}\n'
        s += f'header: {self.header}\n'
        s += f'dir: {self.dir}\n'
        s += f'config_ext: {self.config_ext}\n'
        return s
        
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
        
        
    def run_single(self, main_folder, test_name, sim_name, variables = {}, copy = True, cores = 16): # TODO 
        # XXX Work in progress
        # main_folder = 'Baseline'
        # # test_name   = 'vel'
        # # sim_name    = 'v40'
        # test_name   = 'time'
        # sim_name    = '1cGrif'
        
            
        # Intialize simulation runner
        proc = Simulation_runner()
        
        # Build sheet 
        builder = config_builder(self.mat)
        png_file = builder.save_view(self.config_path, 'sheet')
        builder.add_pullblocks()
        lammps_file_txt, lammps_file_info = builder.save_lammps("sheet", ext = self.config_ext, path = self.config_path)
        config_data = f'sheet_{self.config_ext}'
        proc.add_variables(config_data = config_data)
        
        # Directories 
        proc.config_path = self.config_path
        
        
        header = f"egil:{main_folder}/{test_name}/"
        dir = f"{header}{sim_name}/"
    
        if copy:
            proc.move_files_to_dest(["../friction_simulation/setup_sim.in", 
                            "../friction_simulation/stretch.in",
                            "../friction_simulation/drag.in",
                            "../potentials/si.sw",
                            "../potentials/C.tersoff",
                            f"../config_builder/{proc.variables['config_data']}.txt",
                            f"../config_builder/{proc.variables['config_data']}_info.in" ], header)
        
        sim = Simulator(directory = dir, overwrite=True)
        sim.copy_to_wd( "../friction_simulation/friction_procedure.in")
            
        # proc.variables["out_ext"] = sim_name
        sim.set_input_script("../friction_simulation/friction_procedure.in", **proc.variables)
        slurm_args = {'job-name':sim_name, 'partition':'normal', 'ntasks':cores, 'nodes':1}
        sim.run(num_procs=cores, lmp_exec="lmp", slurm=True, slurm_args=slurm_args)
                  
    
    def run_multi(self, F_N, variables = {}, num_procs = 16):
        
        # Intialize simulation runner
        proc = Simulation_runner(variables)
        
        # Build sheet 
        builder = config_builder(self.mat)
        png_file = builder.save_view(self.config_path, 'sheet')
        builder.add_pullblocks()
        lammps_file_txt, lammps_file_info = builder.save_lammps("sheet", ext = self.config_ext, path = self.config_path)
        config_data = f'sheet_{self.config_ext}'
        proc.add_variables(config_data = config_data)
        
        # Directories 
        proc.config_path = self.config_path
        
        # Multi run settings 
        # num_stretch_files = 1
        # F_N = np.array([5])*1e-9
        # # F_N = np.sort(np.random.uniform(0.1, 10, 10))*1e-9
        
        
        # proc.add_variables(num_stretch_files = num_stretch_files, 
        #                    RNSEED = '$RANDOM',
        #                    run_rupture_test = 1,
        #                    stretch_max_pct = 2.0,
        #                    root = '.',
        #                    dump_freq = 10000)
        
        
        # Start multi run
        root_path = proc.multi_run(self.header, self.dir, F_N, num_procs = num_procs, jobname = self.config_ext)
        
        # Transfer config npy- and png-file 
        proc.move_files_to_dest([self.npy_file, png_file], root_path)
       
        # Remove generated files locally
        os.remove(png_file)
        os.remove(lammps_file_txt)
        os.remove(lammps_file_info)
        


def run_files(filenames, header, simname):
    for file in filenames:
        gen = data_generator(file, header, simname)
        gen.run_multi()
        


if __name__ == "__main__":
    # run_files(get_files_in_folder('../config_builder/nocut_sizes/', exclude = 'DS_Store'), header =  'egil:CONFIGS/nocut_sizes')
    
    # files = get_files_in_folder('../config_builder/honeycomb/', ext = '.npy')
    # files = files[20:]
    # print(files)
    # run_files(files, header =  'egil:CONFIGS/honeycomb', simname = 'single_run')
    
    pass
    
    
   