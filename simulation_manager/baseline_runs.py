from simulation_runner import *


def single_run():
    main_folder = 'Baseline'
    test_name   = 'vel'
    sim_name    = 'v40'
    
    variables = {
        "dt": 0.001,
        "T": 100.0, # [K]
        "relax_time": 15,
        "pause_time1": 5,
        "pause_time2": 5,
        "stretch_speed_pct": 0.005,
        "stretch_max_pct": 0,
        "drag_length": 200 ,
        "drag_speed": 20, # [m/s]
        "K": 30.0,
        "drag_dir_x": 0,
        "drag_dir_y": 1,
        "F_N": 1e-9, # [N]
        "config_data": "sheet_nocut_108x113",
        "root": "..",
        "out_ext": sim_name, 
        "run_rupture_test": 0
    }
    
    proc = Simulation_runner(variables)
    header = f"egil:{main_folder}/{test_name}/"
    dir = f"{header}{sim_name}/"
    
    proc.move_files_to_dest(["../friction_simulation/setup_sim.in", 
                        "../friction_simulation/stretch.in",
                        "../friction_simulation/drag.in",
                        "../potentials/si.sw",
                        "../potentials/C.tersoff",
                        # "../potentials/CH.airebo",
                        # "../potentials/FeAu-eam-LJ.fs",
                        f"../config_builder/{proc.variables['config_data']}.txt",
                        f"../config_builder/{proc.variables['config_data']}_info.in" ], header)
        
    sim = Simulator(directory = dir, overwrite=True)
    sim.copy_to_wd( "../friction_simulation/friction_procedure.in")
        
    # proc.variables["out_ext"] = sim_name
    sim.set_input_script("../friction_simulation/friction_procedure.in", **proc.variables)
    slurm_args = {'job-name':sim_name, 'partition':'normal', 'ntasks':16, 'nodes':1}
    sim.run(num_procs=16, lmp_exec="lmp", slurm=True, slurm_args=slurm_args)


def vary_variable(test_name = 'dt', variable_name = 'dt', variable_values = [], sim_prefix = None):
    assert len(variable_values) > 0, "Variable values has length 0."
    main_folder = 'Baseline'
    if sim_prefix is None:
        sim_prefix = test_name
    
    # Default values
    variables = {
        "dt": 0.001,
        "T": 100.0, # [K]
        "relax_time": 15,
        "pause_time1": 5,
        "pause_time2": 5,
        "stretch_speed_pct": 0.005,
        "stretch_max_pct": 0,
        "drag_length": 400 ,
        "drag_speed": 20, # [m/s]
        "K": 30.0,
        "drag_dir_x": 0,
        "drag_dir_y": 1,
        "F_N": 1e-9, # [N]
        "config_data": "sheet_nocut_108x113",
        "root": "..",
        "out_ext": 'tmp', 
        "run_rupture_test": 0
    }
    

    proc = Simulation_runner(variables)
    header = f"egil:{main_folder}/{test_name}/"
    
    proc.move_files_to_dest(["../friction_simulation/setup_sim.in", 
                             "../friction_simulation/stretch.in",
                             "../friction_simulation/drag.in",
                             "../potentials/si.sw",
                             "../potentials/C.tersoff"], header)
    
    if variable_name != "config_data":
        proc.move_files_to_dest([f"../config_builder/{proc.variables['config_data']}.txt",
                                 f"../config_builder/{proc.variables['config_data']}_info.in" ], header)
  
  
  
    for i, val in enumerate(variable_values):
        if hasattr(sim_prefix, '__len__'):
            sim_name  = sim_prefix[i]
        else:
            sim_name  = f'{sim_prefix}_{val}'
        dir = f"{header}{sim_name}/"
        
        
        proc.variables["out_ext"] = sim_name
        proc.variables[variable_name] = val
        
        if variable_name == "config_data":
            proc.move_files_to_dest([f"../config_builder/{proc.variables['config_data']}.txt",
                                    f"../config_builder/{proc.variables['config_data']}_info.in" ], header)
     
        sim = Simulator(directory = dir, overwrite=True)
        sim.copy_to_wd( "../friction_simulation/friction_procedure.in")
            
        sim.set_input_script("../friction_simulation/friction_procedure.in", **proc.variables)
        slurm_args = {'job-name':sim_name, 'partition':'normal', 'ntasks':16, 'nodes':1}
        sim.run(num_procs=16, lmp_exec="lmp", slurm=True, slurm_args=slurm_args)


if __name__ == "__main__":
    
    dt_range = [0.00025, 0.0005, 0.001, 0.002]
    vel_range = [1, 5, 10, 20, 30, 40]
    temp_range = [5, 50, 100, 200, 300, 400, 500]
    K_range = [0, 10, 30, 50, 100]
    size_name = ['64x62', '86x87', '108x113', '130x138', '152x163', '174x189']
    size_range = [f"sheet_nocut_{s}" for s in size_name]
    

    # vary_variable(test_name = 'size', variable_name = 'config_data',  variable_values = size_range, sim_prefix = size_name)
    # drag_length()