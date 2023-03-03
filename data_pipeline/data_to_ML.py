import os
import numpy as np

import csv
import shutil



import sys
sys.path.append('../') # parent folder: MastersThesis
from analysis.analysis_utils import *


def locate_multi_dir(dir):
    target = "config.png"
    multi_dirs = []
    for root, dirs, files in os.walk(dir):
        if len(dirs) == 0: continue # Ignore deepest subdirs
        if target in files:
            multi_dirs.append(root)
    if len(multi_dirs) == 0:
        print(f'multi_dirs is empty.')
    return multi_dirs  
        
    

def convert_data(multi_dirs, dest):
    info_file = 'info_file.txt'
    friction_ext = 'Ff.txt'
    mean_pct = 0.5
    std_pct = 0.35
    

    count = 0
    
    os.mkdir(dest)
    for dir in multi_dirs:
        # Find config array
        for file in os.listdir(dir):
            if '.npy' == file[-4:]:
                config_file = file
                break
                
        # Analyse data points
        for root, dirs, files in os.walk(dir, topdown=False):
            if len(dirs) > 0: continue # Only use lowest subdirs
            print(f'\rAnalysing file | count = {count+1:05d} | dir = {root} ', end = '')
            data = {}
            
        
            # Create folder for data point
            subdest = f"{dest}/sim_{count:05d}"
            os.mkdir(subdest)
            
            # Move config png and npy til folder
            shutil.copyfile(os.path.join(dir, 'config.png'), 
                            os.path.join(subdest, 'config.png'))
            shutil.copyfile(os.path.join(dir, config_file), 
                            os.path.join(subdest, 'config.npy'))
            
            
            
            # Read info dict
            info_dict = read_info_file(os.path.join(root, info_file))
            
            data['stretch'] = info_dict['SMAX']
            data['F_N'] = metal_to_SI(info_dict['F_N'], 'F')*1e9
            data['scan_angle'] = (info_dict['drag_dir_x'], info_dict['drag_dir_y'])
            data['dt'] = info_dict['dt']
            data['T'] = info_dict['T']
            data['drag_speed'] = info_dict['drag_speed']
            data['drag_length'] = info_dict['drag_length']
            data['K'] = info_dict['K']
            data['stretch_speed_pct'] = info_dict['stretch_speed_pct']
            data['relax_time'] = info_dict['relax_time']
            data['pause_time1'] = info_dict['pause_time1']
            data['pause_time2'] = info_dict['pause_time2']
                
            # Read rupture test
            rupture_test_dict = read_info_file(os.path.join(dir, 'rupture_test.txt'))
            data['rupture_stretch'] = rupture_test_dict['rupture_stretch']
            data['is_ruptured'] = int(info_dict['is_ruptured'])
            
            # Analyse friction file
            if not info_dict['is_ruptured']:
                friction_file = find_single_file(root, ext = friction_ext)     
                _, fricData = analyse_friction_file(friction_file, mean_pct, std_pct)
                
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
            
            # Read matrix and calculate porosity (void fraction)   
            mat = np.load(os.path.join(dir, config_file))
            porosity = np.sum(mat < 0.5)/np.sum(mat)
            data['porosity'] = porosity
            
            
            # Write data to file in csv format
            with open (os.path.join(subdest,'val.csv'), 'w') as csvfile:  
                w = csv.writer(csvfile)
                for key, val in data.items():
                    w.writerow([key, val])
            
        

            count += 1
            
            
    print()





if __name__ == "__main__":
    # dir = '../Data/CONFIGS/honeycomb'
    # dest = '../Data/ML_data/honeycomb'
    # dir = '../Data/CONFIGS/popup'
    # dest = '../Data/ML_data/popup'
    
    dir = '../Data/Baseline_fixmove/nocut/multi_stretch'
    dest = '../Data/ML_data/nocut'
    
    
    
    
    multi_dirs = locate_multi_dir(dir)
    convert_data(multi_dirs, dest)
    pass
    
    
    
    
    
 