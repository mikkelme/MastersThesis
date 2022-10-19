import numpy as np
# from lammps_simulator import Simulator

import sys
sys.path.append('../../lammps-simulator_ssh') # parent folder: MastersThesis
from lammps_simulator.simulator import Simulator
from lammps_simulator.device import Device
import subprocess

class Friction_procedure:
    def __init__(self, variables):
        # Standard variables
        self.variables = {
            "dt": 0.001,
            "config_data": "sheet_substrate",
            "relax_time": 5,
            "stretch_speed_pct": 0.05,
            "stretch_max_pct": 0.2,
            "pause_time1": 5,
            "F_N": 10e-9, # [N]
            "pause_time2": 5,
            "drag_dir_x": 0,
            "drag_dir_y": 1,
            "drag_speed": 1, # [m/s]
            "drag_length": 30 ,
            "K": 30.0,
            "root": ".",
            "out_ext": "_default" # put date here
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
       
def move_file_to(file, dest):
    subprocess.run(['rsync', '-av', '-mkpath', file, dest])
    # subprocess.run(['rsync', '-av', file, self.ssh + ':' + self.wd + tail]) 
    

def great4_runner():
    
    # Reference settings for NG4
    # variables = { 
    # "dt": 0.001, 
    # "relax_time": 5,
    # "stretch_speed_pct": 0.05,
    # "pause_time1": 5,
    # "F_N": 10e-9, # [N]
    # "pause_time2": 5,
    # "drag_dir_x": 0,
    # "drag_dir_y": 1,
    # "drag_speed": 1, # [m/s]
    # "drag_length": 30,
    # "K": 30.0,
    # "root": "..",
    # }
    
    # Changeable
    variables = { 
    "dt": 0.001, 
    "relax_time": 5,
    "stretch_speed_pct": 0.05,
    "pause_time1": 5,
    "F_N": 10e-9, # [N]
    "pause_time2": 5,
    "drag_dir_x": 0,
    "drag_dir_y": 1,
    "drag_speed": 1, # [m/s]
    "drag_length": 30,
    "K": 0,
    "root": "..",
    }
    


    

    proc = Friction_procedure(variables)

    # header = "NewGreat4/" 
    header = "egil:NewGreat4_K0/"
    common_files = ["../friction_simulation/setup_sim.in", 
                    "../friction_simulation/friction_procedure.in",
                    "../potentials/si.sw",
                    "../potentials/CH.airebo",
                    ]

    for file in common_files:
        move_file_to(file, header)
        

    extentions = ["nocut_nostretch", "nocut_20stretch", "cut_nostretch", "cut_20stretch"]
    config_data = ["sheet_substrate_nocuts", "sheet_substrate_nocuts", "sheet_substrate", "sheet_substrate"]
    stretch_max_pct = [0.0, 0.2, 0.0, 0.2]
    
    exit() # Safety break
    for i, ext in enumerate(extentions):
        dir = header + ext
        sim = Simulator(directory = dir, overwrite=True)
        sim.copy_to_wd( "../friction_simulation/run_friction_sim.in",
                        f"../config_builder/{config_data[i]}.txt",
                        f"../config_builder/{config_data[i]}_info.in"
                        )
        
        proc.variables["out_ext"] = '_' + ext
        proc.variables["config_data"] = config_data[i]
        proc.variables["stretch_max_pct"] = stretch_max_pct[i]
        sim.set_input_script("../friction_simulation/run_friction_sim.in", **proc.variables)
        sim.create_subdir("output_data")
        
        slurm_args = {'job-name':'NG4_K0', 'partition':'normal', 'ntasks':16, 'nodes':1}
        sim.run(num_procs=16, lmp_exec="lmp", slurm=True, slurm_args=slurm_args)
        # sim.run(num_procs=1, lmp_exec="lmp_mpi")

        # mpirun -n 1 lmp_mpi -in run_friction_sim.in -var dt 0.001 -var config_data sheet_substrate -var relax_time 1 -var stretch_speed_pct 0.05 -var stretch_max_pct 0.0 -var pause_time1 1 -var F_N 0.4993207256 -var pause_time2 0 -var drag_dir_x 0 -var drag_dir_y 1 -var drag_speed 0.05 -var drag_length 1 -var K 1.8724527210000002 -var root .. -var out_ext default


def multi_run(sim, proc, config_data, num_stretch_files, F_N, num_procs = 16, jobname = 'MULTI'):
    sim.copy_to_wd( "../friction_simulation/setup_sim.in",
                    f"../config_builder/{config_data}.txt",
                    f"../config_builder/{config_data}_info.in",
                    "../potentials/si.sw",
                    "../potentials/CH.airebo",
                    "../friction_simulation/drag_from_restart_file.in"
                    )
    
    sim.set_input_script("../friction_simulation/stretch_with_reset_files.in", num_stretch_files = num_stretch_files, **proc.variables)    
    slurm_args = {'job-name':jobname, 'partition':'normal', 'ntasks':num_procs, 'nodes':1}
    sim.pre_generate_jobscript(num_procs=num_procs, lmp_exec="lmp", slurm_args = slurm_args)    

    proc.variables['root'] = '../..'
    job_array = 'job_array=('
    for i in range(len(F_N)):
        proc.variables['F_N'] = F_N[i]
        proc.convert_units(["F_N"])
        sub_exec_list = Device.get_exec_list(num_procs = num_procs, 
                                             lmp_exec = "lmp", 
                                             lmp_args = {'-in': '../../drag_from_restart_file.in'}, 
                                             lmp_var = proc.variables | {'out_ext':'_tmp'})
        job_array += '\n\n\"'
        job_array += Device.gen_jobscript_string(sub_exec_list, slurm_args, linebreak = False)
        job_array += '\"'
    job_array += ')'
    
    sim.add_to_jobscript(f"\nwait\n\
    \n{job_array}\n\
    \nfor file in *.restart; do\
    \n    [ -f \"$file\" ] || break\
    \n    folder1=\"${{file%.*}}\"_folder\
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
    
    

def one_config_multi_data():
    
    variables = { 
    "dt": 0.001, 
    "relax_time": 5,
    "stretch_speed_pct": 0.05,
    "stretch_max_pct": 0.2,
    "pause_time1": 5,
    "F_N": 10e-9, # [N]
    "pause_time2": 5,
    "drag_dir_x": 0,
    "drag_dir_y": 1,
    "drag_speed": 5, # [m/s]
    "drag_length": 30,
    "K": 30.0,
    "root": ".",
    }
    
    
    proc = Friction_procedure(variables)
    
    # Variables 
    num_stretch_files = 5
    F_N = [10e-9, 20e-9, 30e-9]
    config_data = "sheet_substrate" 
    
    dir = "egil:one_config_multi_data"
    
    sim = Simulator(directory = dir, overwrite=True)
    multi_run(sim, proc, config_data, num_stretch_files, F_N, num_procs = 16, jobname = 'MULTI')

    
   
    
    
    
    

    


if __name__ == "__main__":
    # great4_runner()
    one_config_multi_data()