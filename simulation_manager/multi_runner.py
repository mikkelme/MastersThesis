### Initial scripts for running simulations on the cluster

import sys
sys.path.append('../') # parent folder: MastersThesis
from simulation_manager.simulation_runner import *

def one_config_multi_data(header, dir, variables, F_N):
    proc = Simulation_runner(variables)
   

    proc.variables["out_ext"] = sim_name
    proc.multi_run(header, dir, F_N, num_procs = 1, jobname = jobname)
    
    


if __name__ == "__main__":
    main_folder = 'CapacityTest'
    test_name   = 'test'
    sim_name    = 'test'
    jobname     = 'CapTest' 
    
    header = f"egil:{main_folder}/{test_name}/"
    dir = f"{header}{sim_name}"
    
    variables = {
        "num_stretch_files": 1,
        "stretch_max_pct": 0.01,
        "config_data": "sheet_cut_108x113",
        "dump_freq": 0,
        "relax_time": 1,
    }
  
    # Varying paramters  
    # F_N = np.array([10, 100, 200, 300])*1e-9
    F_N = np.linspace(0.1e-9, 1e-9, 100)
    
    one_config_multi_data(header, dir, variables, F_N)
    