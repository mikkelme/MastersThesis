from simulation_runner import *

def one_config_multi_data():
    main_folder = 'ConfigMulti'
    test_name   = 'nocuts'
    sim_name    = 'ref1'
    
    
    variables = {
        "dt": 0.001,
        "temp": 300.0, # [K]
        "relax_time": 5,
        "pause_time1": 5,
        "pause_time2": 5,
        "stretch_speed_pct": 0.001,
        "drag_speed": 1, # [m/s]
        "drag_length": 200,
        "K": 0, #30.0,
        "root": "..",
        "out_ext": date.today(), 
        "config_data": "sheet_substrate_nocuts",
        "stretch_max_pct": 0.0,
        "drag_dir_x": 0,
        "drag_dir_y": 1,
        "F_N": 100e-9, # [N]
    }
  
    # Varying paramters  
    num_stretch_files = 5
    F_N = np.linspace(10e-9, 200e-9, 3)
    
  
    proc = Simulation_runner(variables)
    header = f"egil:{main_folder}/{test_name}/"
    dir = f"{header}{sim_name}"
    
    proc.variables["out_ext"] = sim_name
    proc.multi_run(header, dir, num_stretch_files, F_N, num_procs = 16, jobname = sim_name)


if __name__ == "__main__":
     one_config_multi_data()