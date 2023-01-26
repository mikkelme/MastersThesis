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
    fig_max, fig_mean = variable_dependency(folders, names, 'T', '$T$ [K]', default = 300)
    if save:
        fig_max.savefig("../article/figures/baseline/variables_temp_max.pdf", bbox_inches="tight")
        fig_mean.savefig("../article/figures/baseline/variables_temp_mean.pdf", bbox_inches="tight")

    
    
    
def vel(path, save = False):
    common_folder = 'vel' 
    folders = [os.path.join(path, 'nocut', common_folder), 
               os.path.join(path, 'popup', common_folder),
               os.path.join(path, 'honeycomb', common_folder)]
    names = ['nocut', 'popup', 'honeycomb']
    convert = metal_to_SI(1, 's')/metal_to_SI(1,'t')
    fig_max, fig_mean = variable_dependency(folders, names, 'drag_speed', 'Drag speed [m/s]', convert = convert, default = 20)
    if save:
        fig_max.savefig("../article/figures/baseline/variables_vel_max.pdf", bbox_inches="tight")
        fig_mean.savefig("../article/figures/baseline/variables_vel_mean.pdf", bbox_inches="tight")

    
    
    
def spring(path, save = False):
    common_folder = 'spring' 
    folders = [os.path.join(path, 'nocut', common_folder), 
               os.path.join(path, 'popup', common_folder),
               os.path.join(path, 'honeycomb', common_folder)]
    names = ['nocut', 'popup', 'honeycomb']
    convert = metal_to_SI(1, 'F')/metal_to_SI(1,'s')
    fig_max, fig_mean = variable_dependency(folders, names, 'K', '$K$ [N/m]', convert = convert, map = {0: 120/convert}, default = 0)
    ax_mean = fig_mean.axes[0]
    
        
    xtick_labels = ax_mean.get_xticks()
    xtick_labels[-2] = 'inf'
    ax_mean.set_xticklabels(xtick_labels)
    
    
    x_min, x_max = ax_mean.get_xlim()
    ticks = [(tick - x_min)/(x_max - x_min) for tick in ax_mean.get_xticks()]
    line_pos = (ticks[-3] + ticks[-2])/2
    
    ylen = 0.015  
    xdis = 0.005
    kwargs = dict(transform=ax_mean.transAxes, color='k', clip_on=False)
    ax_mean.plot((line_pos - xdis, line_pos - xdis), (-ylen, +ylen), **kwargs)     
    ax_mean.plot((line_pos + xdis, line_pos + xdis), (-ylen, +ylen), **kwargs)     
    
    
    if save:
        fig_max.savefig("../article/figures/baseline/variables_spring_max.pdf", bbox_inches="tight")
        fig_mean.savefig("../article/figures/baseline/variables_spring_mean.pdf", bbox_inches="tight")

    
    
    
def dt(path, save = False):
    common_folder = 'dt' 
    folders = [os.path.join(path, 'nocut', common_folder), 
               os.path.join(path, 'popup', common_folder),
               os.path.join(path, 'honeycomb', common_folder)]
    names = ['nocut', 'popup', 'honeycomb']
    convert = 1e3 # ps -> fs
    fig_max, fig_mean = variable_dependency(folders, names, 'dt', '$dt$ [fs]', convert = convert, default = 1)
    if save:
        fig_max.savefig("../article/figures/baseline/variables_dt_max.pdf", bbox_inches="tight")
        fig_mean.savefig("../article/figures/baseline/variables_dt_mean.pdf", bbox_inches="tight")

    
    

def variable_dependency(folders, names, variable_key, xlabel, convert = None, error = 'both', map = None, default = None):
    mean_window_pct = 0.5 # relative length of the mean window [% of total duration]
    std_window_pct = 0.2  # relative length of the std windoe [% of mean window]
    
    
    fig_mean = plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    ax_mean = plt.gca()
    

    fig_max = plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    ax_max = plt.gca()
    
    linestyle = '-'
    marker = 'o'
    
    line_and_marker = {'linestyle': '-', 
                    'marker': 'o'}
    
    for i, folder in enumerate(folders):
        files = get_files_in_folder(folder, ext = '_Ff.txt')
        num_files = len(files)
        
        var = np.full(num_files, np.nan)    
        Ff_max = np.full(num_files, np.nan)   
        Ff_mean = np.full(num_files, np.nan)   
        Ff_mean_std = np.full(num_files, np.nan)
        for j, file in enumerate(files):
            info, data = analyse_friction_file(file, mean_window_pct, std_window_pct)    
            if (info, data) == (None, None):
                continue
            
            
            var[j] = float(info[variable_key])
            Ff_max[j] = data['Ff'][0, 0]
            Ff_mean[j] = data['Ff'][0, 1]
            Ff_mean_std[j] = data['Ff_std'][0]*data['Ff'][0, 1] # rel. error -> abs. error
            
            if map is not None:
                if var[j] in map:
                    var[j] = map[var[j]]
        
        sort = np.argsort(var)
        var = var[sort]
        Ff_max = Ff_max[sort]
        Ff_mean = Ff_mean[sort]
        Ff_mean_std = Ff_mean_std[sort]
        
        if convert is not None:
            var *= convert
    
        
        
        colors = [color_cycle(0), color_cycle(1), color_cycle(3)]
        
        color_and_label = {'color': colors[i], 'label': names[i]}
        
        
        # Max friction 
        ax_max.plot(var, Ff_max, **line_and_marker, color = colors[i], label = names[i]) 
        
        
        # Mean friction
        if error == 'bar':
            ax_mean.errorbar(var, Ff_mean, yerr = Ff_mean_std, **line_and_marker, **color_and_label, capsize=6) 
        elif error == 'shade':
            ax_mean.plot(var, Ff_mean, **line_and_marker, **color_and_label) 
            ax_mean.fill_between(var, Ff_mean + Ff_mean_std, Ff_mean - Ff_mean_std, alpha = 0.1, color = color_and_label['color'])
        elif error == 'both':
            ax_mean.errorbar(var, Ff_mean, yerr = Ff_mean_std, **line_and_marker, **color_and_label, capsize=6) 
            ax_mean.fill_between(var, Ff_mean + Ff_mean_std, Ff_mean - Ff_mean_std, alpha = 0.1, color = color_and_label['color'])

        else:
            exit(f'error display, error = {error}, is not defied,')
            
            
    if default is not None:
        for ax in [ax_max, ax_mean]:
            vline(ax, default, linestyle = '--', color = 'black', linewidth = 1, zorder = 0, label = "Default")
       
        
    
    ax_mean.set_xlabel(xlabel, fontsize=14)
    ax_mean.set_ylabel(r'$\langle F_\parallel \rangle$ [nN]', fontsize=14)
    ax_mean.legend(fontsize = 13)#, fancybox = True, shadow = True)
    fig_mean.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    ax_max.set_xlabel(xlabel, fontsize=14)
    ax_max.set_ylabel(r'$\max \ F_\parallel$ [nN]', fontsize=14)
    ax_max.legend(fontsize = 13)
    fig_max.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    return fig_max, fig_mean



def tmp(path, save = False):
    common_folder = 'multi_stretch' 
    folders = [os.path.join(path, 'nocut', common_folder), 
               os.path.join(path, 'popup', common_folder),
               os.path.join(path, 'honeycomb', common_folder)]
    names = ['nocut', 'popup', 'honeycomb']
    
    print(folders)
    print(names)


if __name__ == "__main__":
    
    # path = '../Data/Baseline'
    # temp(path, save = False)
    
    path = '../Data/Baseline_fixmove'
    # temp(path, save = False)
    # vel(path, save = False)
    # spring(path, save = False)
    # dt(path, save = False)
    
    
    tmp(path)
    
    plt.show()