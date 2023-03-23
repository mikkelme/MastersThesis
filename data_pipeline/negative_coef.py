from data_generator import *


def multi_coupling_popup():
    """ Run multiple FN for coupling simulation with configuration of choice """
    name = 'popup'
    file = '../config_builder/baseline/pop1_7_5.npy'   
    # simname = f'multi_coupling_{name}' 
    simname = f'multi_coupling_free_{name}' 
    gen = Data_generator(file, header = f'egil:negative_coef', simname = simname, config_ext = name)
    variables = {'num_stretch_files': 100, 
                    'RNSEED'           : -1,
                    'run_rupture_test' : 1,
                    "stretch_speed_pct": 0.001,
                    'F_N'              : 15e-9,
                    'R'                : 6,
                    "stretch_max_pct"  : 0.3,
                    'root'             : '.',
                    'dump_freq'        : 100000}

    F_N = np.array([0])*1e-9
    
    # manual_coupling_drag.in
    # gen.run_multi(F_N, variables, num_procs_initial = 16, num_procs = 4, partition = 'normal', scripts = ["manual_coupling_stretch.in", "manual_coupling_drag.in"])

    # manual_coupling_free_drag.in
    variables['K'] = 1e4 # Stiff springs to approximate fix move conditions
    gen.run_multi(F_N, variables, num_procs_initial = 16, num_procs = 4, partition = 'normal', scripts = ["manual_coupling_stretch.in", "manual_coupling_free_drag.in"])
    
    

def multi_coupling_honeycomb():
    """ Run multiple FN for coupling simulation with configuration of choice """
    name = 'honeycomb'
    file = '../config_builder/baseline/hon3215.npy'   
    # simname = f'multi_coupling_{name}'
    # simname = f'multi_coupling_free_{name}'
    simname = f'multi_coupling_free_{name}_zoom'
    gen = Data_generator(file, header = f'egil:negative_coef', simname = simname, config_ext = name)
    # variables = {'num_stretch_files': 100, 
    #                 'RNSEED'           : -1,
    #                 'run_rupture_test' : 1,    # 0
    #                 "stretch_speed_pct": 0.0001,
    #                 'F_N'              : 10e-9,  # 1.5e-9
    #                 'R'                : 6,
    #                 "stretch_max_pct"  : 1.5, # 1.0
    #                 'root'             : '.',
    #                 'dump_freq'        : 0}
    variables = {'num_stretch_files': 1000, 
                    'RNSEED'           : -1,
                    'run_rupture_test' : 0,
                    "stretch_speed_pct": 0.0001,
                    'F_N'              : 1.2e-9,
                    'R'                : 6,
                    "stretch_max_pct"  : 1.0,
                    'root'             : '.',
                    'dump_freq'        : 0}

    F_N = np.array([0])*1e-9
    
    # gen.run_multi(F_N, variables, num_procs = 4, partition = 'mini', scripts = ["manual_coupling_stretch.in", "manual_coupling_drag.in"])
    # gen.run_multi(F_N, variables, num_procs_initial = 16, num_procs = 4, partition = 'normal', scripts = ["manual_coupling_stretch.in", "manual_coupling_drag.in"])
    
    # manual_coupling_free_drag.in
    variables['K'] = 1e4 # Stiff springs to approximate fix move conditions
    gen.run_multi(F_N, variables, num_procs_initial = 16, num_procs = 4, partition = 'normal', scripts = ["manual_coupling_stretch.in", "manual_coupling_free_drag.in"])
        
        
        
if __name__ == '__main__':
    # multi_coupling_popup()
    # multi_coupling_honeycomb()
    pass