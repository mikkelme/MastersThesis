import numpy as np



class Friction_procedure:
    def __init__(self, root, script_path, config_path, out_path, output_file = "commands1.in", setup_file = "friction_simulation.in", simulation_action = "std_friction_procedure_spring.in"):
        # Load input
        self.setup_file = setup_file
        self.simulation_action = simulation_action

        # Set lammps include path (using lammps variables)
        self.path = "${root}${script_path}/"
        
        # --- Begin procedure file --- #
        self.output_file = output_file   
        self.outfile = open(output_file, 'w')

        # Set path
        self.outfile.write(f"variable root string {root}\n")
        self.outfile.write(f"variable script_path string {script_path}\n")
        self.outfile.write(f"variable config_path string {config_path}\n")
        self.outfile.write(f"variable out_path string {out_path}\n")
     
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
        self.F_N = 0.8e-9  # [N]

        # Wait before dragging sheet
        self.pause_time2 = 1

        # Drag sheet
        self.drag_dir_x = 0
        self.drag_dir_y = 1


        self.drag_speed = 5.0  # [m/s]

        self.drag_length = 10 # [Å]
        self.K = 30.0  # spring constant: (N/m)

        self.out_ext = "1"

        # --- Convert: SI -> metal --- #
        self.convert_units(["F_N", "drag_speed", "K"])

        # --- Define variables and initialise value holder 
        self.varname = ["relax_time", "stretch_speed_pct", "stretch_max_pct", "pause_time1", "F_N", "pause_time2", "drag_dir_x", "drag_dir_y", "drag_speed", "drag_length", "K", "out_ext"]
        self.varval = [np.nan for varname in self.varname]



    def __str__(self):
        string = "----- VARIABLES -----\n"
        for i, varname in enumerate(self.varname):
            string += f"{varname} = {eval('self.'+varname, {'self': self})}\n"
        string += "--------------------"
        return string


    def convert_units(self, varnames):
        convertion_dict = {     "F_N": self.N_to_eV_over_ang, 
                                "drag_speed": self.m_to_ang/self.s_to_ps, 
                                "K": self.N_to_eV_over_ang/self.m_to_ang}

        for varname in varnames:
            try:
                conv = convertion_dict[varname]
            except KeyError:
                print(f"KeyError: No convertion for\"{varname}\"")
                continue
            
            new_val = eval(f"self.{varname}*{conv}", {'self': self})
            exec(f"self.{varname} = {new_val}")

       

    def add_run(self):
        """ Write variables changes and script start commands to file """
        changes = False

        self.outfile.write("\n")
        self.outfile.write("clear\n")
        self.outfile.write(f"include {self.path}{self.setup_file}\n")
        for i, varname in enumerate(self.varname):
            varval = eval('self.'+varname, {'self': self})
            update = varval != self.varval[i]
            if update:
                changes = True
                self.varval[i] = varval
                if isinstance(varval, str):
                    self.outfile.write(f"variable {varname} string {varval}\n")
                else:
                    self.outfile.write(f"variable {varname} equal {varval}\n")

        assert changes, "No variables changes made"
        self.outfile.write(f"include {self.path}{self.simulation_action}\n")

                  

def different_drag_speeds():
    # Path 
    root = "/Users/mikkelme/Documents/Github/MastersThesis"
    script_path = "/friction_experiment"
    config_path = "/config_builder"
    out_path = "${script_path}/output_data"



    # Settings
    proc = Friction_procedure(  root, script_path, config_path, out_path,
                                output_file = "drag_speed_commands.in", 
                                setup_file = "friction_simulation.in", 
                                simulation_action = "std_friction_procedure_spring.in"
                                )


    proc.relax_time = 0.5           # [ps] 
    proc.stretch_speed_pct = 0.2    # [% of pattern ylen per picoseconds]
    proc.stretch_max_pct = 0.1
    proc.pause_time1 = 0.5
    proc.F_N = 0.08e-9              # [eV/Å]
    proc.pause_time2 = 1            # [ps]
    proc.drag_dir_x = 0
    proc.drag_dir_y = 1
    proc.drag_length = 10           # [Å]
    proc.K = 30.0                   # [N/m]
    proc.convert_units(["F_N", "K"])


    # Systematic change
    drag_speeds = [1, 2, 5, 10, 50]
    # proc.relax_time = 0
    # proc.stretch_max_pct = 0
    # proc.pause_time2 = 1
    # proc.drag_length = 10
    # drag_speeds = [500, 1000]
    for drag_speed in drag_speeds:
        proc.drag_speed = drag_speed
        proc.out_ext = f"\"_v{drag_speed}\""
        proc.convert_units(["drag_speed"])
        proc.add_run()


    proc.outfile.close()



def custom():
   # Path 
    root = "/Users/mikkelme/Documents/Github/MastersThesis"
    script_path = "/friction_experiment"
    config_path = "/config_builder"
    out_path = "${script_path}/output_data"



    # Settings
    proc = Friction_procedure(  root, script_path, config_path, out_path,
                                output_file = "stretch_no_stretch.in", 
                                setup_file = "friction_simulation_periodic.in", 
                                simulation_action = "std_friction_procedure_spring.in"
                                )


    proc.relax_time = 5           # [ps] 
    proc.stretch_speed_pct = 0.05    # [% of pattern ylen per picoseconds]
    proc.stretch_max_pct = 0.2
    proc.pause_time1 = 1.0
    proc.F_N = 2*0.08e-9              # [eV/Å]
    proc.pause_time2 = 5.0            # [ps]
    proc.drag_dir_x = 0
    proc.drag_dir_y = 1
    proc.drag_length = 5           # [Å]
    proc.drag_speed = 5.0           # [m/s]
    proc.K = 30.0                   # [N/m]
    proc.convert_units(["F_N", "K", "drag_speed"])

    proc.out_ext = "2xFN_stretch"
    proc.add_run()

    
    proc.stretch_max_pct = 0
    proc.out_ext = "2xFN_nostretch"
    proc.add_run()



    proc.outfile.close()




# std_friction_procedure_spring.in



if __name__ == "__main__":
    # different_drag_speeds()
    custom()