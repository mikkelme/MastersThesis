from data_generator import *

def baseline(files):
    for file in files:
        gen = Data_generator(file, 'header', 'simname')
        gen.run_single(main_folder, test_name, sim_name, variables, copy = True, cores = 16)
    
    

def baseline_multi_stretch(names, files):
    """ Vary stretch for 3 different normal forces """
    for i in range(len(files)):
        name, file = names[i], files[i]
        gen = Data_generator(file, header = f'egil:Baseline_fixmove/{name}', simname = 'multi_stretch', config_ext = name)
        variables = {'num_stretch_files': 30, 
                     'RNSEED'           : -1,
                     'run_rupture_test' : 1,
                     'stretch_max_pct'  : 2.0,
                     'root'             : '.',
                     'dump_freq'        : 10000}

        F_N = np.array([0.1, 1, 10])*1e-9
        # F_N = np.sort(np.random.uniform(0.1, 10, 10))*1e-9
        
        gen.run_multi(F_N, variables, num_procs = 16)
        
        
def baseline_multi_FN(names, files):
    """ Vary F_N for 5 different stretch (relative to rupture stretch of each configuration) """
    
    SMAX = [1.0, 0.33, 0.16]
    for i in range(len(files)):
        name, file = names[i], files[i]
        gen = Data_generator(file, header = f'egil:Baseline_fixmove/{name}', simname = 'multi_FN', config_ext = name)
        variables = {'num_stretch_files': 5, 
                     'RNSEED'           : -1,
                     'run_rupture_test' : 0,
                     'stretch_max_pct'  : SMAX[i],
                     'root'             : '.',
                     'dump_freq'        : 10000}

        
        F_N = np.logspace(-1, 2, 30)*1e-9
        gen.run_multi(F_N, variables, num_procs = 16)
        

def baseline_multi_FN_lin():
    """ Vary F_N linearly for no stretch nocut sheet """
    
    name = 'nocut'
    file = '../config_builder/baseline/nocut.npy'
    gen = Data_generator(file, header = f'bigfacet:Baseline_fixmove/{name}', simname = 'multi_FN_lin_even', config_ext = name)
    variables = {   'num_stretch_files': 1, 
                    'RNSEED'           : -1,
                    'run_rupture_test' : 0,
                    'stretch_max_pct'  : 0,
                    'root'             : '.',
                    'dump_freq'        : 0}
        
    F_N = np.linspace(0.1, 10, 32)*1e-9
    gen.run_multi(F_N, variables, num_procs = 1)
 
    
    

def baseline_temp(names, files):
    """ Vary temperature """
    variable_key = 'T'
    test_name = 'temp3'
    # temp_range = [5, 50, 100, 200, 300, 400, 500]
    # sim_names = ['T5', 'T50', 'T100', 'T200', 'T300', 'T400', 'T500']
    
    temp_range = np.linspace(10, 500, 50).astype('int')
    sim_names = [f'T{T}' for T in temp_range]
    vary_variable(names, files, test_name, sim_names, variable_key, temp_range)


def baseline_vel(names, files):
    """ Vary drag velocity """
    variable_key = 'drag_speed'
    test_name = 'vel2'
    # vel_range = [1, 5, 10, 20, 30, 50, 100]
    # sim_names = ['v1', 'v5', 'v10', 'v20', 'v30', 'v50', 'v100']
    vel_range = np.arange(2, 100+1).astype('int')
    sim_names = [f'v{v}' for v in vel_range]
    
    vary_variable(names, files, test_name, sim_names, variable_key, vel_range)
    
    
def baseline_dt(names, files):
    """ Vary dt """
    variable_key = 'dt'
    test_name = 'dt2'
    # dt_range = [0.0001, 0.00025, 0.0005, 0.001, 0.0015, 0.002]
    # sim_names = ['dt01', 'dt025', 'dt05', 'dt10', 'dt15', 'dt20' ]
    dt_range = np.linspace(0.0001, 0.002, 39)
    sim_names = [f'dt{dt*10**5:03.0f}' for dt in dt_range]
    
    vary_variable(names, files, test_name, sim_names, variable_key, dt_range)

def baseline_K(names, files):
    """ Vary spring constant """
    variable_key = 'K'
    test_name = 'spring'
    # K_range = [0, 1, 5, 20, 30, 50, 100]
    # sim_names = ['K0', 'K1', 'K5', 'K20', 'K30', 'K50', 'K100']
    K_range = np.linspace(0, 200, 41).astype('int')
    sim_names = [f'K{K}' for K in K_range]
    
    vary_variable(names, files, test_name, sim_names, variable_key, K_range)
    
    
    
def vary_variable(names, files, test_name, simnames, variable_key, variable_range):
    num_procs = 16
    stretch = 0
    
    for i in range(len(files)):
        name, file = names[i], files[i]
        for j, val in enumerate(variable_range):    
            gen = Data_generator(file, header = f'egil:Baseline_fixmove/{name}/{test_name}', simname = simnames[j], config_ext = name)
            variables = {variable_key       : val,
                         'dump_freq'        : 0,
                         'stretch_max_pct'  : stretch}

            gen.run_single(variables, num_procs = num_procs, copy = j==0)
            # gen.run_single(variables, num_procs = num_procs, copy = False)
 


if __name__ == '__main__':
    # names = ['honeycomb', 'nocut', 'popup']
    # files = get_files_in_folder('../config_builder/baseline/', ext = '.npy')
    # print(names)
    # print(files)
    
    # baseline_temp(names, files)
    # baseline_vel(names, files)
    # baseline_K(names, files)
    # baseline_dt(names, files)
    
    # baseline_multi_stretch(names, files)
    # baseline_multi_FN(names, files)
    # baseline_multi_FN_lin()
    
    
    # filenames = get_files_in_folder('../config_builder/baseline/', ext = 'npy')
    # variables = {'dump_freq': 10000,
    #              'stretch_speed_pct': 0.001,
    #              'T': 300}
    
    
    # # Honeycomb
    # file = filenames[0]
    # variables['stretch_max_pct'] = 1.40
    # gen = Data_generator(file, header = f'bigfacet:Baseline_fixmove/honeycomb/contact', simname = 'hon_contact', config_ext = 'contact')
    # gen.run_single(variables, num_procs = 16, copy = True)
    
    # #Popup
    # file = filenames[1]
    # variables['stretch_max_pct'] = 0.25
    # gen = Data_generator(file, header = f'bigfacet:Baseline_fixmove/popup/contact', simname = 'pop_contact', config_ext = 'contact')
    # gen.run_single(variables, num_procs = 16, copy = True)
    
    # # Nocut
    # file = filenames[2]
    # variables['stretch_max_pct'] = 0.40
    # gen = Data_generator(file, header = f'bigfacet:Baseline_fixmove/nocut/contact', simname = 'nocut_contact', config_ext = 'contact')
    # gen.run_single(variables, num_procs = 16, copy = True)
    
    
   
   
    
    pass