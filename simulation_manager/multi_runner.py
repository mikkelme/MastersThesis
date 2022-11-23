from simulation_runner import *

def one_config_multi_data():
    main_folder = 'Multi'
    test_name   = 'nocuts'
    sim_name    = 'ref1'
    jobname     = 'nocuts' 
    
    
    variables = {
        "dt": 0.001,
        "T": 300.0, # [K]
        "relax_time": 5,
        "pause_time1": 5,
        "pause_time2": 5,
        "stretch_speed_pct": 0.001,
        "drag_speed": 5, # [m/s]
        "drag_length": 100,
        "K": 0, #30.0,
        "root": "..",
        "out_ext": date.today(), 
        "config_data": "sheet_substrate_nocuts",
        "stretch_max_pct": 0.25,
        "drag_dir_x": 0,
        "drag_dir_y": 1,
        "F_N": 100e-9, # [N]
    }
  
    # Varying paramters  
    num_stretch_files = 6
    F_N = np.linspace(10e-9, 190e-9, 3)
    
    proc = Simulation_runner(variables)
    header = f"egil:{main_folder}/{test_name}/"
    dir = f"{header}{sim_name}"
    
    proc.variables["out_ext"] = sim_name
    proc.multi_run(header, dir, num_stretch_files, F_N, num_procs = 16, jobname = jobname)


if __name__ == "__main__":
    #  one_config_multi_data()
    pass