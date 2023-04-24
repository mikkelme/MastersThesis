import numpy as np
import os

import sys
sys.path.append('../') # parent folder: MastersThesis
from config_builder.build_config import *
from simulation_manager.multi_runner import *
from analysis.analysis_utils import get_files_in_folder

class Data_generator:
    def __init__(self, filename, header =  'egil:CONFIGS/TEST', simname = 'test', config_ext = None):
        
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
        self.simname = simname
        
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
        
        
    def run_single(self, variables = {}, num_procs = 16, copy = True): 
        # Intialize simulation runner
        proc = Simulation_runner(variables)
        
        
        # Directories 
        # proc.config_path = self.config_path
        
        config_data = f'sheet_{self.config_ext}'
        proc.add_variables(config_data = config_data)
    
        if copy:
            # Build sheet 
            builder = config_builder(self.mat)
            png_file = builder.save_view(self.config_path, 'sheet')
            builder.add_pullblocks()
            lammps_file_txt, lammps_file_info = builder.save_lammps("sheet", ext = self.config_ext, path = self.config_path)

            # Move files to header
            proc.move_files_to_dest(["../friction_simulation/setup_sim.in", 
                        "../friction_simulation/stretch.in",
                        "../friction_simulation/drag.in",
                        "../potentials/si.sw",
                        "../potentials/C.tersoff",
                        f"{self.config_path}/{proc.variables['config_data']}.txt",
                        f"{self.config_path}/{proc.variables['config_data']}_info.in" ], self.header)
    
        
        
        sim = Simulator(directory = self.dir, overwrite=True)
        main_script = "../friction_simulation/friction_procedure.in"
        sim.copy_to_wd(main_script)
            
        proc.variables["out_ext"] = self.simname
        sim.set_input_script(main_script, **proc.variables)
        slurm_args = {'job-name':self.simname, 'partition':'normal', 'ntasks':num_procs, 'nodes':1}
        sim.run(num_procs=num_procs, lmp_exec="lmp", slurm=True, slurm_args=slurm_args)
          
        if copy:  
            # Transfer config npy- and png-file 
            
            proc.move_files_to_dest([self.npy_file, png_file], self.header)
        
            # Remove generated files locally
            os.remove(png_file)
            os.remove(lammps_file_txt)
            os.remove(lammps_file_info)    
                
    
    def run_multi(self, F_N = None, variables = {}, num_procs_initial = None, num_procs = 16, scripts = None, partition = 'normal'):
        
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
        # num_stretch_files = 15
        # num_stretch_files = variables['num_stretch_files']
        
        # if F_N is None:
        #     F_N = np.sort(np.random.uniform(0.1, 10, 3))*1e-9
        
        
        # proc.add_variables(num_stretch_files = num_stretch_files, 
        #                    RNSEED = '$RANDOM',
        #                    run_rupture_test = 1,
        #                    stretch_max_pct = 2.0,
        #                    root = '.',
        #                    dump_freq = 0)
        
        
        # Start multi run
        root_path = proc.multi_run(self.header, self.dir, F_N, num_procs_initial, num_procs = num_procs, jobname = self.config_ext, scripts = scripts, partition = partition)
        
        # Transfer config npy- and png-file 
        proc.move_files_to_dest([self.npy_file, png_file], root_path)
       
        # Remove generated files locally
        os.remove(png_file)
        os.remove(lammps_file_txt)
        os.remove(lammps_file_info)
        


def run_files(filenames, header, simname, num_procs_initial = None, num_procs = 16):
    
    variables = {   'num_stretch_files': 15, 
                    'RNSEED': '$RANDOM',
                    'run_rupture_test': 1,
                    'stretch_max_pct': 2.0,
                    'root': '.',
                    'dump_freq': 0
                }
    
    variables['num_stretch_files'] = 30
    variables['RNSEED'] = -1
    for file in filenames:
        # F_N = np.sort(np.random.uniform(0.1, 10, 3))*1e-9
        F_N = np.array([5])*1e-9
        gen = Data_generator(file, header, simname)
        gen.run_multi(F_N = F_N, variables = variables, num_procs_initial = num_procs_initial, num_procs = num_procs)
        


if __name__ == "__main__":
    # filenames = get_files_in_folder('../config_builder/popup/', ext = 'npy')
    # filenames = get_files_in_folder('../config_builder/honeycomb/', ext = 'npy')
    # filenames = get_files_in_folder('../config_builder/RW/', ext = 'npy')
    
    
    filenames = get_files_in_folder('../ML/RW_search', ext = 'npy')
    # run_files(filenames, header =  'egil:CONFIGS/RW_search_test', simname = 'test', num_procs_initial = 16, num_procs = 4)
   
    
    
   