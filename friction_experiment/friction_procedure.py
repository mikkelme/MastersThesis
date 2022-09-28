import numpy as np



class friction_procedure():
    def __init__(self, param_file = "procedure1.in", setup_file = "friction_simulation.in", std_procedure = "std_friction_procedure_spring.in"):

        path = "./"

        # --- Begin procedure file --- #
        self.param_file = param_file   
        self.outfile = open(param_file, 'w')
        self.outfile.write(f"include {path}{setup_file}")

        # --- Convertion factors: SI -> metal --- #
        self.N_to_eV_over_ang = 6.24150907e8    # force: N -> eV/Å
        self.m_to_ang = 1e10                    # distance: m -> Å
        self.s_to_ps = 1e12                     # time: s -> ps


        # --- Standard values --- #
        # Relax time
        self.relax_time = 0.5 # [ps] 

        # Stretch sheet
        self.stretch_speed_pct = 0.2    # [% of pattern ylen per picoseconds]
        self.stretch_max_pct = 0.1

        # Wait before applying normal force
        self.pause_time1 = 0.5

        # Apply normal force 
        self.F_N = 0.08e-9  # [eV/Å]

        # Wait before dragging sheet
        self.pause_time2 = 1

        # Drag sheet
        self.drag_dir = np.array([1, 0])
        self.drag_speed = 5.0  # [m/s]

        self.drag_length = 10 # [Å]
        self.K = 30.0  # spring constant: (N/m)


        # --- Convert: SI -> metal --- #
        self.F_N *= self.N_to_eV_over_ang               # [N] -> [eV/Å]
        self.drag_speed *= self.m_to_ang/self.s_to_ps   # [m/s] -> [Å/ps]
        self.K *= self.N_to_eV_over_ang/self.m_to_ang   # [N/m] -> [eV/Å^2]



# std_friction_procedure_spring.in



if __name__ == "__main__":
    procedure = friction_procedure()



    procedure.outfile.close()