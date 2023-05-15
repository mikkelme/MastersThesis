### Class for handling the submission of 
### simulation jobs for various simulation procedures


import numpy as np
from datetime import date
import subprocess
import os

# from lammps_simulator import Simulator

import sys
sys.path.append('../../lammps-simulator_ssh') 
from lammps_simulator.simulator import Simulator
from lammps_simulator.device import Device
from lammps_simulator.device import SlurmGPU

class Simulation_runner:
    def __init__(self, variables = {}) :
        # Default variables
        self.variables = {
            "dt": 0.001,
            "T": 300, # [K]
            "relax_time": 15,
            "pause_time1": 5,
            "pause_time2": 5,
            "stretch_speed_pct": 0.01,
            "stretch_max_pct": 0.2,
            "run_rupture_test": 0,
            "num_stretch_files": 0,
            "RNSEED": -1,
            "drag_length": 400,
            "drag_speed": 20, # [m/s]
            "K": 0,
            "drag_dir_x": 0,
            "drag_dir_y": 1,
            "F_N": 1e-9, # [N]
            "config_data": "sheet_substrate",
            "root": "..",
            "out_ext": date.today(),
            "dump_freq": 10000
        }
        

        self.config_path = '../config_builder'
        
        # --- Convertion factors: SI -> metal --- #
        self.N_to_eV_over_ang = 6.24150907e8    # force: N -> eV/Å
        self.m_to_ang = 1e10                    # distance: m -> Å
        self.s_to_ps = 1e12                     # time: s -> ps
        
        # Dict for specific convertions 
        self.conv_dict = {    
            "F_N": self.N_to_eV_over_ang, 
            "drag_speed": self.m_to_ang/self.s_to_ps, 
            "K": self.N_to_eV_over_ang/self.m_to_ang }

        # Convert default values before updating with input
        self.convert_units(["F_N", "K", "drag_speed"]) 
        self.update_variables(variables)
        
        
    def update_variables(self, dict):
        # --- Update variables in class dict --- #
        for key in dict:
            if key in self.variables:
                self.variables[key] = dict[key]
                if key in self.conv_dict:
                    self.convert_units([key])
            else: 
                print(f"WARNING: Variable \"{key}\" is not expected")
                self.variables[key] = dict[key]
                
    def add_variables(self, **kwargs):
        self.update_variables(kwargs)
        
    def convert_units(self, varnames):
        for key in varnames:
            try:
                conv = self.conv_dict[key]
            except KeyError:
                print(f"KeyError: No convertion for \"{key}\"")
                continue
            
            self.variables[key] *= conv 
       
    
    def __str__(self):
        s = "# Class variable dictionary\n"
        for key in self.variables:
            s += f'{key} = {self.variables[key]}\n'
        s += f'\n# Class variables\n'
        s += f'config path = {self.config_path}'
        return s
    
    
    def move_file_to_dest(self, file, dest):
        if '(' in file or ')' in file:
            file = file.replace("(", "\(")
            file = file.replace(")", "\)")
        
        ssh, dir = dest.split(':')
        subprocess.run(f'rsync -av --rsync-path=\"mkdir -p {dir} && rsync\" {file} {ssh}:{dir}', shell = True)
        
    
    def move_files_to_dest(self, files, dest):
        for file in files:
            self.move_file_to_dest(file, dest)
            
    
        
    def multi_run(self, header, dir, F_N, num_procs_initial = None, num_procs = 16, jobname = 'MULTI', scripts = None, partition = 'normal'):
        print_info = True # Print ({stretch}, {F_N}) sets
        
        # Abbreviation (avoid calling self.variables a lot)
        num_stretch_files = self.variables['num_stretch_files']
        stretch_max_pct = self.variables['stretch_max_pct']
        RNSEED = self.variables['RNSEED']
        
        
        nice_val = 100
        # nice_val = None
        if nice_val is None:
            nice = ''
        else:
            nice = f'--nice {nice_val}'
        
        # Verify validity of RNSEED and interpret for print info
        if RNSEED == '$RANDOM':
            RN_stretch = True
        else:
            try:
                RNSEED = int(RNSEED)
                self.variables['RNSEED'] = RNSEED
                RN_stretch = RNSEED >= 0
    
            except ValueError:
                exit(f'RNSEED: {RNSEED}, is not valid')
                
        # Print info
        if print_info:
            print(f"Samples (stretch x F_N): {num_stretch_files } x {len(F_N)} = {num_stretch_files * len(F_N)}")
            if RN_stretch:
                print("Stretch: Uniform random in intervals:")
                stretch_step =  stretch_max_pct/num_stretch_files 
                for i in range(num_stretch_files - 1):
                    print(f"[{i*stretch_step:g}, {(i+1)*stretch_step:g}),", end = " ")
                print(f"[{(num_stretch_files - 1)*stretch_step:g}, {(num_stretch_files)*stretch_step:g})")
            else:
                print(f"Stretch: {np.around(np.linspace(0, stretch_max_pct, num_stretch_files), decimals = 3)}")
            print(f"F_N: {F_N*1e9} nN")
        # exit("Safety break")
    
    
        # Make directory and transfer scripts    
        sim = Simulator(directory = dir, overwrite=False) 
        dir = sim.sim_settings['dir'] # Get updated dir (relevant for overwrite = False)
        root_path = os.path.normpath(os.path.join(dir, self.variables['root']))
        
        
        stretch_script = "stretch.in"
        drag_script = "drag.in"
        if scripts is not None:
            stretch_script = scripts[0]
            drag_script = scripts[1]
        
        self.move_files_to_dest(["../friction_simulation/setup_sim.in", 
                        f"../friction_simulation/{stretch_script}",
                        f"../friction_simulation/{drag_script}",
                        "../potentials/si.sw",
                        "../potentials/C.tersoff",
                        f"{self.config_path}/{self.variables['config_data']}.txt",
                        f"{self.config_path}/{self.variables['config_data']}_info.in" ], root_path)
    
        
        # Set input script and slurm arguments
        if num_procs_initial is None:
            num_procs_initial = num_procs
        sim.set_input_script(f"../friction_simulation/{stretch_script}", **self.variables)    
        # slurm_args = {'job-name':jobname, 'partition':partition, 'ntasks':num_procs_initial, 'nodes':1}
        slurm_args = {'job-name':jobname, 'partition':partition, 'ntasks':num_procs_initial, 'cpus-per-task':1}
        
        
        # --- Jobscript for multi run --- #
        sim.pre_generate_jobscript(num_procs=num_procs_initial, lmp_exec="lmp", slurm_args = slurm_args)    

        # Set parameters for stretch 
        # slurm_args = {'job-name':jobname, 'partition':partition, 'ntasks':num_procs, 'nodes':1}
        slurm_args = {'job-name':jobname, 'partition':partition, 'ntasks':num_procs, 'cpus-per-task':1}
        
        self.variables['root'] += '/../..'
        job_array = 'job_array=('
        for i in range(len(F_N)):
            self.variables['F_N'] = F_N[i]
            self.convert_units(["F_N"])
            sub_exec_list = Device.get_exec_list(num_procs = num_procs, 
                                                 lmp_exec = "lmp", 
                                                 lmp_args = {'-in': self.variables['root']+f'/{drag_script}'}, 
                                                 lmp_var = self.variables | {'out_ext':'drag'}) 
            job_array += '\n\n\"'
            job_array += Device.gen_jobscript_string(sub_exec_list, slurm_args, linebreak = False)
            job_array += '\"'
        job_array += ')'
        
        # Add bash commands for executing drag.in for all stretch restart files
        sim.add_to_jobscript(f"\nwait\
        \n{job_array}\n\
        \nfor file in *_restart; do\
        \n    [ -f \"$file\" ] || break\
        \n    folder1=\"${{file%_*}}\"_folder\
        \n    mkdir $folder1\
        \n    mv $file $folder1/$file\
        \n    cd $folder1\
        \n    for i in ${{!job_array[@]}}; do\
        \n      folder2=job\"$i\"\
        \n      mkdir $folder2\
        \n      echo \"${{job_array[$i]}} -var restart_file ../$file\" > $folder2/job$i.sh\
        \n      cd $folder2\
        \n      sbatch job$i.sh {nice}\
        \n      cd ..\
        \n    done\
        \n    cd ..\
        \ndone")
        
        # --- RUN --- #
        sim.run(slurm = True, execute = True)
        return root_path
        
            

if __name__ == "__main__":
    
    # test = Simulation_runner()
    # test.move_file_to_dest("../config_builder/baseline/sp1(7,5).npy", "egil:./") 
    pass