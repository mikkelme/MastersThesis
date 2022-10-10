import numpy as np
from lammps_simulator import Simulator


class Friction_procedure:
    def __init__(self, variables):
        # Standard variables
        self.variables = {
            "dt": 0.001,
            "config_data": "data.txt",
            "relax_time": 5,
            "stretch_speed_pct": 0.05,
            "stretch_max_pct": 0.2,
            "pause_time1": 5,
            "F_N": 0.8e-9, # [N]
            "pause_time2": 5,
            "drag_dir_x": 0,
            "drag_dir_y": 1,
            "drag_speed": 5, # [m/s]
            "drag_length": 10 ,
            "K": 30.0
        }
        
        # --- Convertion factors: SI -> metal --- #
        self.N_to_eV_over_ang = 6.24150907e8    # force: N -> eV/Å
        self.m_to_ang = 1e10                    # distance: m -> Å
        self.s_to_ps = 1e12                     # time: s -> ps
        
        # Dict for specific convertions 
        self.conv_dict = {    
            "F_N": self.N_to_eV_over_ang, 
            "drag_speed": self.m_to_ang/self.s_to_ps, 
            "K": self.N_to_eV_over_ang/self.m_to_ang
        }

        
        self.update_variables(variables)
        self.convert_units(["F_N", "K", "drag_speed"])
        
    def update_variables(self, variables):
        """ Update variables in class dict"""
        for key in variables:
            if key in self.variables:
                self.variables[key] = variables[key]
            else: 
                print(f"WARNING: Variable \"{key}\" is not defined")
    
            
    def convert_units(self, varnames):
        for key in varnames:
            try:
                conv = self.conv_dict[key]
            except KeyError:
                print(f"KeyError: No convertion for \"{key}\"")
                continue
            
            self.variables[key] *= conv 
       


def test_runner():
    
    variables = { "dt": 0.001,
                "config_data": "sheet_substrate",
                "relax_time": 1,
                "stretch_speed_pct": 0.05,
                "stretch_max_pct": 0.2,
                "pause_time1": 1,
                "F_N": 0.8e-9, # [N]
                "pause_time2": 1,
                "drag_dir_x": 0,
                "drag_dir_y": 1,
                "drag_speed": 5, # [m/s]
                "drag_length": 10 ,
                "K": 30.0
            }

  
    proc = Friction_procedure(variables)
    
    
    sim = Simulator(directory = "simulation", overwrite=True)
    sim.copy_to_wd("../potentials/si.sw", 
                   "../potentials/CH.airebo", 
                   "../friction_experiment/setup_sim.in",
                   "../friction_experiment/friction_procedure.in",
                   "../config_builder/sheet_substrate.txt",
                   "../config_builder/sheet_substrate_info.in"
                   )
    sim.set_input_script("../friction_experiment/run_friction_sim.in", **proc.variables)
    sim.run(num_procs=2, lmp_exec="lmp_mpi")


    


if __name__ == "__main__":
    test_runner()
    # different_drag_speeds()