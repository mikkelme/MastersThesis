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
            "temp": 100.0, # [K]
            "relax_time": 5,
            "pause_time1": 5,
            "pause_time2": 5,
            "stretch_speed_pct": 0.05,
            "drag_speed": 1, # [m/s]
            "drag_length": 30 ,
            "K": 30.0,
            "root": ".",
            "out_ext": date.today(), 
            "config_data": "sheet_substrate",
            "stretch_max_pct": 0.2,
            "drag_dir_x": 0,
            "drag_dir_y": 1,
            "F_N": 10e-9, # [N]
        }
        
        
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
       
    def move_file_to(self, file, dest):
        subprocess.run(['rsync', '-av', '-mkpath', file, dest])
        
    # TODO: Modify so this works in new version (class Simulation_runner)
    def multi_run(self, sim, proc, num_stretch_files, F_N, num_procs = 16, jobname = 'MULTI'):
        config_data = proc.variables['config_data']
        sim.copy_to_wd( "../friction_simulation/setup_sim.in",
                        f"../config_builder/{config_data}.txt",
                        f"../config_builder/{config_data}_info.in",
                        "../potentials/si.sw",
                        "../potentials/C.tersoff",
                        # "../potentials/CH.airebo",
                        "../friction_simulation/drag.in"
                        )
        
        
        sim.set_input_script("../friction_simulation/stretch.in", num_stretch_files = num_stretch_files, **proc.variables)    
        slurm_args = {'job-name':jobname, 'partition':'normal', 'ntasks':num_procs, 'nodes':1}
        sim.pre_generate_jobscript(num_procs=num_procs, lmp_exec="lmp", slurm_args = slurm_args)    

        proc.variables['root'] = '../..'
        job_array = 'job_array=('
        for i in range(len(F_N)):
            proc.variables['F_N'] = F_N[i]
            proc.convert_units(["F_N"])
            sub_exec_list = Device.get_exec_list(num_procs = num_procs, 
                                                lmp_exec = "lmp", 
                                                lmp_args = {'-in': '../../drag.in'}, 
                                                lmp_var = proc.variables | {'out_ext':'ext'})
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
        
        sim.run(slurm = True)
        
            

if __name__ == "__main__":
    test = Simulation_runner()
    print(test.variables['out_ext'])