import sys
sys.path.append('../') # parent folder: MastersThesis
import matplotlib.pyplot as plt
import numpy as np
from plot_set import *
from analysis.analysis_utils import *



def temp(path, save = False):
    common_folder = 'temp' 
    folders = [os.path.join(path, 'nocut', common_folder), 
               os.path.join(path, 'popup', common_folder),
               os.path.join(path, 'honeycomb', common_folder)]
    names = ['nocut', 'popup', 'honeycomb']
    variable_dependency(folders, names, 'T', '$T$ [K]', save)
    
    
def vel(path, save = False):
    common_folder = 'vel' 
    folders = [os.path.join(path, 'nocut', common_folder), 
               os.path.join(path, 'popup', common_folder),
               os.path.join(path, 'honeycomb', common_folder)]
    names = ['nocut', 'popup', 'honeycomb']
    convert = metal_to_SI(1, 's')/metal_to_SI(1,'t')
    variable_dependency(folders, names, 'drag_speed', '$Drag speed$ [---]', save, convert = convert)
    
    

def variable_dependency(folders, names, variable_key, xlabel, save = False, convert = None):
    mean_window_pct = 0.5 # relative length of the mean window [% of total duration]
    std_window_pct = 0.2  # relative length of the std windoe [% of mean window]
    
    
    fig_mean = plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    ax_mean = plt.gca()
    

    fig_max = plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    ax_max = plt.gca()
    
    for i, folder in enumerate(folders):
        files = get_files_in_folder(folder, ext = '_Ff.txt')
        num_files = len(files)
        
        var = np.zeros(num_files)    
        Ff_max = np.zeros(num_files)   
        Ff_mean = np.zeros(num_files)   
        Ff_mean_std = np.zeros(num_files)
        for j, file in enumerate(files):
            info, data = analyse_friction_file(file, mean_window_pct, std_window_pct)    
            
            
            var[j] = float(info[variable_key])
            Ff_max[j] = data['Ff'][0, 0]
            Ff_mean[j] = data['Ff'][0, 1]
            Ff_mean_std[j] = data['Ff_std'][0]
            
        
        sort = np.argsort(var)
        var = var[sort]
        Ff_max = Ff_max[sort]
        Ff_mean = Ff_mean[sort]
        Ff_mean_std = Ff_mean_std[sort]
        
        if convert is not None:
            var *= convert
    
        
        
        ax_max.plot(var, Ff_max, '-o', color = color_cycle(i), label = names[i]) 
        ax_mean.errorbar(var, Ff_mean, yerr = Ff_mean_std, marker = 'o', capsize=6, color = color_cycle(i), label = names[i]) 
        
        # ax_mean.plot(var, Ff_mean, '-o', color = color_cycle(i), label = names[i]) 
    
    ax_mean.set_xlabel(xlabel, fontsize=14)
    ax_mean.set_ylabel(r'$\langle F_\parallel \rangle$ [nN]', fontsize=14)
    ax_mean.legend(fontsize = 13)#, fancybox = True, shadow = True)
    fig_mean.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    ax_max.set_xlabel(xlabel, fontsize=14)
    ax_max.set_ylabel(r'$\max \ F_\parallel$ [nN]', fontsize=14)
    ax_max.legend(fontsize = 13)
    fig_max.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    if save:
        pass
    
        



if __name__ == "__main__":
    path = '../Data/Baseline'
    
    # temp(path, save = False)
    vel(path, save = False)
    plt.show()