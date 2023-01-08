import os
import numpy as np

import sys
sys.path.append('../') # parent folder: MastersThesis
from analysis.analysis_utils import *

# Input: Data folder
# Retrive variables and results
# Gather in new file
# Move to desired location


def locate_multi_dir(dir):
    target = "config.png"
    multi_dirs = []
    for root, dirs, files in os.walk(dir):
        if len(dirs) == 0: continue # Ignore deepest subdirs
        if target in files:
            multi_dirs.append(root)
    return multi_dirs  
        
    

def get_data_points(multi_dirs):
    # target_ext = "Ff.txt"
    info_file = 'info_file.txt'
    friction_ext = 'Ff.txt'
    mean_pct = 0.5
    std_pct = 0.2
    
    for dir in multi_dirs:
        # Find config array
        for file in os.listdir(dir):
            if '.npy' == file[-4:]:
                config_file = os.path.join(dir, file)
                break
                
        # Analyse data points
        for root, dirs, files in os.walk(dir, topdown=False):
            if len(dirs) > 0: continue # Only use lowest subdirs
            # test = np.loadtxt(os.path.join(root, 'info_file.txt'), dtype=<class 'string'>)
            
            data = {}
            info_dict = read_info_file(os.path.join(root, info_file))
            
            data['stretch_pct'] = info_dict['SMAX']
            data['F_N'] = metal_to_SI(info_dict['F_N'], 'F')*1e9
            data['scan_angle'] = (info_dict['drag_dir_x'], info_dict['drag_dir_y'])
            data['dt'] = info_dict['dt']
            data['T'] = info_dict['T']
            data['drag_speed'] = info_dict['drag_speed']
            data['drag_length'] = info_dict['drag_length']
                
                      
            if not info_dict['is_ruptured']:
                friction_file = find_single_file(root, ext = friction_ext)     
                fricData = analyse_friction_file(friction_file, mean_pct, std_pct)
                
                data['Ff_max'] = fricData['Ff'][0, 0]
                data['Ff_mean'] = fricData['Ff'][0, 1]
                data['Ff_mean_std'] = fricData['Ff_std'][0]
                data['contact'] = fricData['contact_mean'][0]
                data['contact_std'] = fricData['contact_std'][0]
                
                
                
            else:
                data['Ff_max'] = np.nan
                data['Ff_mean'] = np.nan
                data['Ff_mean_std'] = np.nan
                data['contact'] = np.nan
                data['contact_std'] = np.nan
            
            
            print(data)

            
            exit()
        
        exit()





if __name__ == "__main__":
    dir = '../Data/CONFIGS/cut_sizes'
    multi_dirs = locate_multi_dir(dir)
    get_data_points(multi_dirs)
    # get_data_points(folder)