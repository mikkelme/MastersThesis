from simulation_runner import *


def drag_length():
    main_folder = 'Baseline'
    test_name   = 'drag_length_s200nN'
    sim_name    = 'ref'
    
    variables = {
        "dt": 0.001,
        "T": 100.0, # [K]
        "relax_time": 15,
        "pause_time1": 5,
        "pause_time2": 10,
        "stretch_speed_pct": 0.001,
        "drag_speed": 1, # [m/s]
        "drag_length": 200,
        "K": 30.0,
        "root": "..",
        "out_ext": date.today(), 
        "config_data": "sheet_substrate_nocuts",
        # "config_data": "sheet_substrate_amorph_nocuts",
        "stretch_max_pct": 0.2,
        "drag_dir_x": 0,
        "drag_dir_y": 1,
        "F_N": 200e-9, # [N]
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
        
    proc.variables["out_ext"] = sim_name
    sim.set_input_script("../friction_simulation/friction_procedure.in", **proc.variables)
    slurm_args = {'job-name':sim_name, 'partition':'normal', 'ntasks':16, 'nodes':1}
    sim.run(num_procs=16, lmp_exec="lmp", slurm=True, slurm_args=slurm_args)


def dt():
    main_folder = 'Baseline'
    test_name   = 'dt'
    
    variables = {
        "dt": 0,
        "T": 100.0, # [K]
        "relax_time": 5,
        "pause_time1": 5,
        "pause_time2": 5,
        "stretch_speed_pct": 0.001,
        "drag_speed": 1, # [m/s]
        "drag_length": 200 ,
        "K": 30.0,
        "root": "..",
        "out_ext": date.today(), 
        "config_data": "sheet_substrate_nocuts",
        "stretch_max_pct": 0.0,
        "drag_dir_x": 0,
        "drag_dir_y": 1,
        "F_N": 10e-9, # [N]
    }


    proc = Simulation_runner(variables)
    header = f"egil:{main_folder}/{test_name}/"
    proc.move_files_to_dest(["../friction_simulation/setup_sim.in", 
                        "../friction_simulation/stretch.in",
                        "../friction_simulation/drag.in",
                        "../potentials/si.sw",
                        "../potentials/C.tersoff"], header)
    
    
    
    dt_range = [0.00025, 0.0005, 0.001, 0.002]
    for dt in dt_range:
        sim_name  = f'{test_name}_{dt}'
        dir = f"{header}{sim_name}/"
        
        proc.variables["out_ext"] = sim_name
        proc.variables["dt"] = dt
        proc.variables["drag_length"] = dt/0.001*200
            
        sim = Simulator(directory = dir, overwrite=True)
        sim.copy_to_wd( "../friction_simulation/friction_procedure.in",
                            f"../config_builder/{proc.variables['config_data']}.txt",
                            f"../config_builder/{proc.variables['config_data']}_info.in" )
            
        sim.set_input_script("../friction_simulation/friction_procedure.in", **proc.variables)
        slurm_args = {'job-name':sim_name, 'partition':'normal', 'ntasks':16, 'nodes':1}
        sim.run(num_procs=16, lmp_exec="lmp", slurm=True, slurm_args=slurm_args)


if __name__ == "__main__":
    # drag_length()
    # dt()