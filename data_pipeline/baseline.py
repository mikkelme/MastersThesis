
from data_generator import *

def baseline(files):
    for file in files:
        gen = Data_generator(file, 'header', 'simname')
        gen.run_single(main_folder, test_name, sim_name, variables, copy = True, cores = 16)
    
    

def baseline_multi_stretch(names, files):
    """ Vary stretch for 3 different normal forces """
    for i in range(len(files)):
        name, file = names[i], files[i]
        gen = Data_generator(file, header = f'egil:Baseline/{name}', simname = 'multi_stretch', config_ext = name)
        variables = {'num_stretch_files': 30, 
                     'RNSEED'           : -1,
                     'run_rupture_test' : 1,
                     'stretch_max_pct'  : 2.0,
                     'root'             : '.',
                     'dump_freq'        : 10000}

        F_N = np.array([0.1, 1, 10])*1e-9
        # F_N = np.sort(np.random.uniform(0.1, 10, 10))*1e-9
        
        gen.run_multi(F_N, variables, num_procs = 4)
        
        
def baseline_multi_FN(names, files):
    pass

def baseline_temp(names, files):
    """ Vary temperature """
    variable_key = 'T'
    test_name = 'temp'
    temp_range = [5, 50, 100, 200, 300, 400, 500]
    sim_names = ['T5', 'T50', 'T100', 'T200', 'T300', 'T400', 'T500']
    vary_variable(names, files, test_name, sim_names, variable_key, temp_range)


def baseline_vel(names, files):
    """ Vary drag velocity """
    variable_key = 'drag_speed'
    test_name = 'vel'
    vel_range = [1, 5, 10, 20, 30, 50, 100]
    sim_names = ['v1', 'v5', 'v10', 'v20', 'v30', 'v50', 'v100']
    vary_variable(names, files, test_name, sim_names, variable_key, vel_range)
    
    
def baseline_dt(names, files):
    """ Vary dt """
    variable_key = 'dt'
    test_name = 'dt'
    dt_range = [0.0001, 0.00025, 0.0005, 0.001, 0.0015, 0.002]
    sim_names = ['dt01', 'dt025', 'dt05', 'dt10', 'dt15', 'dt20' ]
    vary_variable(names, files, test_name, sim_names, variable_key, dt_range)

def baseline_K(names, files):
    """ Vary spring constant """
    variable_key = 'K'
    test_name = 'spring'
    # files.pop(0); names.pop(0)
    # print(files)
    # print(names)
    K_range = [0, 1, 5, 20, 30, 50, 100]
    sim_names = ['K0', 'K1', 'K5', 'K20', 'K30', 'K50', 'K100']
    vary_variable(names, files, test_name, sim_names, variable_key, K_range)
    
    
    
def vary_variable(names, files, test_name, simnames, variable_key, variable_range):
    num_procs = 16
    stretch = 0
    
    for i in range(len(files)):
        name, file = names[i], files[i]
        for i, val in enumerate(variable_range):    
            gen = Data_generator(file, header = f'egil:Baseline/{name}/{test_name}', simname = simnames[i], config_ext = name)
            variables = {variable_key       : val,
                         'dump_freq'        : 10000,
                         'stretch_max_pct'  : stretch}

            gen.run_single(variables, num_procs = num_procs, copy= i==0)
 


if __name__ == '__main__':
    names = ['honeycomb', 'nocut', 'popup']
    files = get_files_in_folder('../config_builder/baseline/', ext = '.npy')
    # print(names)
    # print(files)
    
    # baseline_temp(names, files)
    # baseline_vel(names, files)
    # baseline_K(names, files)
    # baseline_dt(names, files)
    

    baseline_multi_stretch(names, files)
    
    pass