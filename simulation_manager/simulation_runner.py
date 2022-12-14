import numpy as np
from datetime import date
import subprocess

# from lammps_simulator import Simulator

import sys
sys.path.append('../../lammps-simulator_ssh') 
from lammps_simulator.simulator import Simulator
from lammps_simulator.device import Device
from lammps_simulator.device import SlurmGPU

class Simulation_runner:
    def __init__(self, variables = {}) :
        # Standard variables
        self.variables = {
            "dt": 0.001,
            "T": 100.0, # [K]
            "relax_time": 15,
            "pause_time1": 5,
            "pause_time2": 5,
            "stretch_speed_pct": 0.005,
            "stretch_max_pct": 0.2,
            "drag_length": 200 ,
            "drag_speed": 20, # [m/s]
            "K": 30.0,
            "drag_dir_x": 0,
            "drag_dir_y": 1,
            "F_N": 1e-9, # [N]
            "config_data": "sheet_substrate",
            "root": "..",
            "out_ext": date.today(), 
            "run_rupture_test": 0
        }
        
        self.config_path = '../config_builder' # NEW XXX
        
        # --- Convertion factors: SI -> metal --- #
        self.N_to_eV_over_ang = 6.24150907e8    # force: N -> eV/Å
        self.m_to_ang = 1e10                    # distance: m -> Å
        self.s_to_ps = 1e12                     # time: s -> ps
        
        # Dict for specific convertions 
        self.conv_dict = {    
            "F_N": self.N_to_eV_over_ang, 
            "drag_speed": self.m_to_ang/self.s_to_ps, 
            "K": self.N_to_eV_over_ang/self.m_to_ang }

        
        # --- Update variables in class dict --- #
        for key in variables:
            if key in self.variables:
                self.variables[key] = variables[key]
            else: 
                print(f"WARNING: Variable \"{key}\" is not defined")
                
        self.convert_units(["F_N", "K", "drag_speed"])
     
           
    def convert_units(self, varnames):
        for key in varnames:
            try:
                conv = self.conv_dict[key]
            except KeyError:
                print(f"KeyError: No convertion for \"{key}\"")
                continue
            
            self.variables[key] *= conv 
       
    def move_file_to_dest(self, file, dest):
        ssh, dir = dest.split(':')
        subprocess.run(f'rsync -av --rsync-path=\"mkdir -p {dir} && rsync\" {file} {ssh}:{dir}', shell = True)
        
    
    def move_files_to_dest(self, files, dest):
        for file in files:
            self.move_file_to_dest(file, dest)
            
    
        
    def multi_run(self, header, dir, num_stretch_files, F_N, RN_stretch = False, num_procs = 16, jobname = 'MULTI'):
        sim = Simulator(directory = dir, overwrite=False)
        
        print_info = True  
        if print_info:
            print(f"Samples: {num_stretch_files} x {len(F_N)} = {num_stretch_files*len(F_N)}")
            if RN_stretch:
                print("Stretch: Uniform random in intervals:")
                Sstep = self.variables['stretch_max_pct']/num_stretch_files
                for i in range(num_stretch_files-1):
                    print(f"[{i*Sstep:g}, {(i+1)*Sstep:g}),", end = " ")
                print(f"[{(i+1)*Sstep:g}, {(i+2)*Sstep:g})")
            else:
                print(f"Stretch: {np.around(np.linspace(0,self.variables['stretch_max_pct'], num_stretch_files), decimals = 3)}")
            print(f"F_N: {F_N*1e9} nN")
        exit("Safety break")
    
        
        self.move_files_to_dest(["../friction_simulation/setup_sim.in", 
                        "../friction_simulation/stretch.in",
                        "../friction_simulation/drag.in",
                        "../potentials/si.sw",
                        "../potentials/C.tersoff",
                        f"{self.config_path}/{self.variables['config_data']}.txt",
                        f"{self.config_path}/{self.variables['config_data']}_info.in" ], header)
    
        

        sim.set_input_script("../friction_simulation/stretch.in", num_stretch_files = num_stretch_files, **self.variables)    
        sim.set_input_script("../friction_simulation/stretch.in", num_stretch_files = num_stretch_files
                                                                  RNSEED = '$RANDOM',
                                                                  run_rupture_test = 1, **self.variables)    
        
        slurm_args = {'job-name':jobname, 'partition':'normal', 'ntasks':num_procs, 'nodes':1}
        sim.pre_generate_jobscript(num_procs=num_procs, lmp_exec="lmp", slurm_args = slurm_args)    

        self.variables['root'] += '/../..'
        job_array = 'job_array=('
        for i in range(len(F_N)):
            self.variables['F_N'] = F_N[i]
            self.convert_units(["F_N"])
            sub_exec_list = Device.get_exec_list(num_procs = num_procs, 
                                                lmp_exec = "lmp", 
                                                lmp_args = {'-in': self.variables['root']+'/drag.in'}, 
                                                lmp_var = self.variables | {'out_ext':'drag'}) 
            job_array += '\n\n\"'
            job_array += Device.gen_jobscript_string(sub_exec_list, slurm_args, linebreak = False)
            job_array += '\"'
        job_array += ')'
        
        
        sim.add_to_jobscript(f"\nwait\
        \n{job_array}\n\
        \nfor file in *_restart; do\
        \n    [ -f \"$file\" ] || break\
        \n    folder1=\"${{file%_*}}\"_folder\
        \n    mkdir $folder1\
        \n    cd $folder1\
        \n    for i in ${{!job_array[@]}}; do\
        \n      folder2=job\"$i\"\
        \n      mkdir $folder2\
        \n      echo \"${{job_array[$i]}} -var restart_file ../$file\" > $folder2/job$i.sh\
        \n      cd $folder2\
        \n      sbatch job$i.sh\
        \n      cd ..\
        \n    done\
        \n    cd ..\
        \n    mv $file $folder1/$file\
        \ndone")
        
        sim.run(slurm = True, execute = True)
        
            

if __name__ == "__main__":
    
    pass
    # test = Simulation_runner()
    # test.move_file_to_dest("./test1.py", "egil:MYTEST/")