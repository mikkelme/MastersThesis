import numpy as np
# from lammps_simulator import Simulator

import sys
sys.path.append('../../lammps-simulator_ssh') # parent folder: MastersThesis
from lammps_simulator import *
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
            "F_N": 160e-9, # [N]
            "pause_time2": 5,
            "drag_dir_x": 0,
            "drag_dir_y": 1,
            "drag_speed": 5, # [m/s]
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
    "K": 30.0,
    "root": "..",
            }
    
   
    # # Quick test 
    # variables = { 
    # "dt": 0.001, 
    # "relax_time": 0.2,
    # "stretch_speed_pct": 0.2,
    # "pause_time1": 0.2,
    # "F_N": 10e-9, # [N]
    # "pause_time2": 0.2,
    # "drag_dir_x": 0,
    # "drag_dir_y": 1,
    # "drag_speed": 5, # [m/s]
    # "drag_length": 0.01,
    # "K": 30.0,
    # "root": "..",
    #         }
    
    

    proc = Friction_procedure(variables)

    # header = "NewGreat4/" 
    header = "egil:NewGreat4/"
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
        # sim.create_subdir("output_data")
        
        slurm_args = {'job-name':'NG4', 'partition':'normal', 'ntasks':16, 'nodes':1}
        sim.run(num_procs=16, lmp_exec="lmp", slurm=True, slurm_args=slurm_args)
        # sim.run(num_procs=1, lmp_exec="lmp_mpi")

        # mpirun -n 1 lmp_mpi -in run_friction_sim.in -var dt 0.001 -var config_data sheet_substrate -var relax_time 1 -var stretch_speed_pct 0.05 -var stretch_max_pct 0.0 -var pause_time1 1 -var F_N 0.4993207256 -var pause_time2 0 -var drag_dir_x 0 -var drag_dir_y 1 -var drag_speed 0.05 -var drag_length 1 -var K 1.8724527210000002 -var root .. -var out_ext default


def one_config_multi_data():
    
    variables = { 
    "dt": 0.001, 
    "relax_time": 5,
    "stretch_speed_pct": 0.05,
    "pause_time1": 5,
    "F_N": 160e-9, # [N] XXX
    "pause_time2": 5, 
    "drag_dir_x": 0,
    "drag_dir_y": 1,
    "drag_speed": 5, # [m/s]
    "drag_length": 30,
    "K": 30.0,
    "root": "..",
            }
    
    
    # Variables 
    F_N = [100e-9, 150e-9, 200e-9]
    num_stretch_files = 3
    
    # dir = "egil:one_config_multi_data"
    dir = "one_config_multi_data"
    
    config_data = "sheet_substrate"
    sim = Simulator(directory = dir, overwrite=True)
    # sim.copy_to_wd( "../friction_simulation/setup_sim.in",
    #                 f"../config_builder/{config_data}.txt",
    #                 f"../config_builder/{config_data}_info.in",
    #                 "../potentials/si.sw",
    #                 "../potentials/CH.airebo",
    #                 )
    
    sim.set_input_script("../friction_simulation/produce_reset_files.in")#, **proc.variables)
    slurm_args = {'job-name':'great4', 'partition':'normal', 'ntasks':16, 'nodes':1}

    sim.pre_generate_jobscript(num_procs=1, lmp_exec="lmp_mpi", slurm_args = slurm_args)    
    sim.add_to_jobscript("\nwait\n\
    \nfor file in *.restart; do\
    \n    [ -f \"$file\" ] || break\
    \n    lmp_serial -in start_from_restart_file.in -var restart_file $file\
    \ndone"
    )
    
  
    # sim.set_run_settings(slurm_args = slurm_args)
    sim.run(write_jobscript = True, slurm = False, execute = False, slurm_args = slurm_args)
    
   
    
    
    
    
    
    
    exit()
    # for i, ext in enumerate(extentions):
    # dir = header + ext
    # sim = Simulator(directory = dir, overwrite=True)
    # sim.copy_to_wd( "../friction_simulation/run_friction_sim.in",
    #                 f"../config_builder/{config_data[i]}.txt",
    #                 f"../config_builder/{config_data[i]}_info.in"
    #                 )
    
    # proc.variables["out_ext"] = '_' + ext
    # proc.variables["config_data"] = config_data[i]
    # proc.variables["stretch_max_pct"] = stretch_max_pct[i]
    # sim.set_input_script("../friction_simulation/run_friction_sim.in", **proc.variables)
    # sim.create_subdir("output_data")
    # # sim.run(num_procs=1, lmp_exec="lmp_mpi")
    
    # slurm_args = {'job-name':'great4', 'partition':'normal', 'ntasks':16, 'nodes':1}
    # sim.run(num_procs=16, lmp_exec="lmp", slurm=True, slurm_args=slurm_args)

    # # mpirun -n 1 lmp_mpi -in ru
    
    


if __name__ == "__main__":
    great4_runner()
    # one_config_multi_data()