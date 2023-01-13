
from data_generator import *

def baseline(files):
    print(files)
    for file in files:
        gen = data_generator(file, 'header', 'simname')
        gen.run_single(main_folder, test_name, sim_name, variables, copy = True, cores = 16)
        # Working here XXX
    
    


if __name__ == '__main__':

    files = get_files_in_folder('../config_builder/baseline/', ext = '.npy')
    baseline(files)