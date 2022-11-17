import numpy as np
# from lammps_simulator import Simulator
from datetime import date



import sys
sys.path.append('../../lammps-simulator_ssh') # parent folder: MastersThesis
from lammps_simulator.simulator import Simulator
from lammps_simulator.device import Device
from lammps_simulator.device import SlurmGPU
import subprocess

class Friction_procedure:
    def __init__(self, variables = {}) :
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
            "out_ext": "default", # put date here
            "temp": 100.0 # [K]
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
    

      
    
    # exit("Safety break is on!") # Safety break
    
    

    proc = Friction_procedure(variables)

    # header = "NewGreat4/" 
    # header = "bigfacet:NG4_GPU/"
    header = "egil:NG4_CPU/"
    common_files = ["../friction_simulation/setup_sim.in", 
                    "../friction_simulation/stretch.in",
                    "../friction_simulation/drag.in",
                    "../potentials/si.sw",
                    "../potentials/C.tersoff",
                    # "../potentials/CH.airebo",
                    ]

    for file in common_files:
        move_file_to(file, header)
        

    extentions = ["nocut_nostretch", "nocut_20stretch", "cut_nostretch", "cut_20stretch"]
    config_data = ["sheet_substrate_nocuts", "sheet_substrate_nocuts", "sheet_substrate", "sheet_substrate"]
    stretch_max_pct = [0.0, 0.2, 0.0, 0.2]
    
    for i, ext in enumerate(extentions):
        dir = header + ext
        sim = Simulator(directory = dir, overwrite=True)
        sim.copy_to_wd( "../friction_simulation/friction_procedure.in",
                        f"../config_builder/{config_data[i]}.txt",
                        f"../config_builder/{config_data[i]}_info.in"
                        )
        
        proc.variables["out_ext"] = ext
        proc.variables["config_data"] = config_data[i]
        proc.variables["stretch_max_pct"] = stretch_max_pct[i]
        sim.set_input_script("../friction_simulation/friction_procedure.in", **proc.variables)
        # sim.create_subdir("output_data")
        
        slurm_args = {'job-name':'NG4_CPU', 'partition':'normal', 'ntasks':16, 'nodes':1}
        sim.run(num_procs=16, lmp_exec="lmp", slurm=True, slurm_args=slurm_args)
        
                   
        # slurm_args = {'job-name':'NG4_GPU', 'partition':'normal'}
        # lmp_args = {'-pk': 'kokkos newton on neigh half'}
        # GPU_device = SlurmGPU(dir = dir, lmp_exec = 'lmp', lmp_args = lmp_args, slurm = True, slurm_args = slurm_args)
        # sim.run(device = GPU_device)


def multi_run(sim, proc, num_stretch_files, F_N, num_procs = 16, jobname = 'MULTI'):
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
      

def one_config_multi_data():
    
    variables = { 
    "dt": 0.001, 
    "relax_time": 5,
    "stretch_speed_pct": 0.01,
    "stretch_max_pct": 0.26,
    "pause_time1": 5,
    "F_N": 10e-9, # [N]
    "pause_time2": 5,
    "drag_dir_x": 1,
    "drag_dir_y": 0,
    "drag_speed": 1, # [m/s]
    "drag_length": 30,
    "K": 30.0,
    "root": ".",
    "config_data": "sheet_substrate"
    }
    
    
    
    proc = Friction_procedure(variables)
    
    # Variables 
    num_stretch_files = 40
    F_N = np.linspace(1e-9, 200e-9, 10)
    

    dir = "egil:BIG_MULTI_Xdrag"
    sim = Simulator(directory = dir, overwrite=True)
    multi_run(sim, proc, num_stretch_files, F_N, num_procs = 16, jobname = 'Xdrag')

    
    
def custom():
    # Reference settings for NG4
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
    "drag_length": 1,
    "K": 30.0,
    "root": "..",
    }
    

    

    proc = Friction_procedure(variables)

    # header = "bigfacet:GPU_perf/"
    header = "egil:CPU_perf/"
    common_files = ["../friction_simulation/setup_sim.in", 
                    "../friction_simulation/stretch.in",
                    "../friction_simulation/drag.in",
                    "../potentials/si.sw",
                    "../potentials/C.tersoff",
                    ]

    for file in common_files:
        move_file_to(file, header)
        

    ext = "cut_20stretch"
    config_data =  "sheet_substrate"
    stretch_max_pct = 0.20
    
    dir = header + ext
    sim = Simulator(directory = dir, overwrite=True)
    sim.copy_to_wd( "../friction_simulation/run_friction_sim.in",
                    f"../config_builder/{config_data}.txt",
                    f"../config_builder/{config_data}_info.in"
                    )
    
    proc.variables["out_ext"] = ext
    proc.variables["config_data"] = config_data
    proc.variables["stretch_max_pct"] = stretch_max_pct
    sim.set_input_script("../friction_simulation/friction_procedure.in", **proc.variables)
    

    slurm_args = {'job-name':'CPU_perf', 'partition':'normal', 'ntasks':16, 'nodes':1}
    sim.run(num_procs=16, lmp_exec="lmp", slurm=True, slurm_args=slurm_args)
        
                   
    # slurm_args = {'job-name':'GPU_perf', 'partition':'normal'}
    # lmp_args = {'-pk': 'kokkos newton on neigh half'}
    # GPU_device = SlurmGPU(dir = dir, lmp_exec = 'lmp', lmp_args = lmp_args, slurm = True, slurm_args = slurm_args)
    # sim.run(device = GPU_device)

    


if __name__ == "__main__":
    test = Friction_procedure()
    print(test.variables['out_ext'])
    # great4_runner()
    # one_config_multi_data()
    # custom()