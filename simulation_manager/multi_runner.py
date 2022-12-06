from simulation_runner import *

def one_config_multi_data():
    main_folder = 'Multi'
    test_name   = 'big'
    sim_name    = 'cuts2'
    jobname     = 'cuts2' 
    
    
    variables = {
        "dt": 0.001,
        "T": 100.0, # [K]
        "relax_time": 15,
        "pause_time1": 5,
        "pause_time2": 10,
        "stretch_speed_pct": 0.001,
        "drag_speed": 20, # [m/s]
        "drag_length": 200, #
        "K": 30.0,
        "root": "..",
        "out_ext": date.today(), 
        "config_data": "sheet_substrate_big",
        # "config_data": "sheet_substrate_amorph",
        # "config_data": "sheet_substrate_nocuts",
        # "config_data": "sheet_substrate",
        "stretch_max_pct": 0.10,
        "drag_dir_x": 0,
        "drag_dir_y": 1
    }
  
    # Varying paramters  
    num_stretch_files = 5
    # F_N = np.array([10, 100, 200, 300])*1e-9
    F_N = np.linspace(0.1e-9, 1e-9, 3)
    
    print(f"Samples: {num_stretch_files} x {len(F_N)} = {num_stretch_files*len(F_N)}")
    print(f"Stretch: {np.around(np.linspace(0,variables['stretch_max_pct'], num_stretch_files), decimals = 3)}")
    print(f"F_N: {F_N*1e9} nN")
    # exit("Safety break")
    
    proc = Simulation_runner(variables)
    header = f"egil:{main_folder}/{test_name}/"
    dir = f"{header}{sim_name}"
    
    proc.variables["out_ext"] = sim_name
    proc.multi_run(header, dir, num_stretch_files, F_N, num_procs = 16, jobname = jobname)


if __name__ == "__main__":
    # Run ref 5 with ultra low FN?
    one_config_multi_data()