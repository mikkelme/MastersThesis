
from data_generator import *

def baseline(files):
    for file in files:
        gen = Data_generator(file, 'header', 'simname')
        gen.run_single(main_folder, test_name, sim_name, variables, copy = True, cores = 16)
    
    

def baseline_multi_stretch(names, files):
    for i in range(len(files)):
        name, file = names[i], files[i]
        gen = Data_generator(file, header = f'egil:Baseline/{name}', simname = 'multi_stretch', config_ext = config_name)
        variables = {'num_stretch_files': 30, 
                     'RNSEED': -1,
                     'run_rupture_test': 1,
                     'stretch_max_pct': 2.0,
                     'root': '.',
                     'dump_freq': 10000}

        F_N = np.array([0.1, 1, 10])*1e-9
        # F_N = np.sort(np.random.uniform(0.1, 10, 10))*1e-9
        
        gen.run_multi(F_N, variables, num_procs = 16)
        
        
# def baseline_multi_FN

def baseline_temp(names, files):
    for i in range(len(files)):
        name, file = names[i], files[i]
        gen = Data_generator(file, header = f'egil:Baseline/{name}', simname = 'temp', config_ext = config_name)
        # variables = {'num_stretch_files': 30, 
        #              'RNSEED': -1,
        #              'run_rupture_test': 1,
        #              'stretch_max_pct': 2.0,
        #              'root': '.',
        #              'dump_freq': 10000}

        # F_N = np.array([0.1, 1, 10])*1e-9
        # # F_N = np.sort(np.random.uniform(0.1, 10, 10))*1e-9
        
        # gen.run_multi(F_N, variables, num_procs = 16)



# def baseline_vel
# def baseline_dt
# def baseline_K

if __name__ == '__main__':
    config_names = ['honeycomb', 'nocut', 'popup']
    files = get_files_in_folder('../config_builder/baseline/', ext = '.npy')

    print(config_names)
    print(files)
    
    
    # baseline_multi_stretch(config_names, files)
    pass