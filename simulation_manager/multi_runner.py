import sys
sys.path.append('../') # parent folder: MastersThesis
from simulation_manager.simulation_runner import *

def one_config_multi_data(header, dir, variables, F_N):
    proc = Simulation_runner(variables)
   
    
    # print(f"Samples: {num_stretch_files} x {len(F_N)} = {num_stretch_files*len(F_N)}")
    # if RN_stretch:
    #     print("Stretch: Uniform random in intervals:")
    #     Sstep = proc.variables['stretch_max_pct']/num_stretch_files
    #     for i in range(num_stretch_files-1):
    #         print(f"[{i*Sstep:g}, {(i+1)*Sstep:g}),", end = " ")
    #     print(f"[{(i+1)*Sstep:g}, {(i+2)*Sstep:g})")
    # else:
    #     print(f"Stretch: {np.around(np.linspace(0,proc.variables['stretch_max_pct'], num_stretch_files), decimals = 3)}")
    # print(f"F_N: {F_N*1e9} nN")
    # exit("Safety break")
    
    
    
    proc.variables["out_ext"] = sim_name
    proc.multi_run(header, dir, F_N, num_procs = 4, jobname = jobname)
    
    


if __name__ == "__main__":
    main_folder = 'CapacityTest'
    test_name   = 'test'
    sim_name    = 'test'
    jobname     = 'CapTest' 
    
    header = f"egil:{main_folder}/{test_name}/"
    dir = f"{header}{sim_name}"
    
    variables = {
        "num_stretch_files": 2,
        "stretch_max_pct": 0.01,
        "config_data": "sheet_cut_108x113",
        "dump_freq": 0
    }
  
    # Varying paramters  
    # F_N = np.array([10, 100, 200, 300])*1e-9
    F_N = np.linspace(0.1e-9, 1e-9, 25)
    
    one_config_multi_data(header, dir, variables, F_N)
    