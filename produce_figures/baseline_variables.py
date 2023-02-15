import sys
sys.path.append('../') # parent folder: MastersThesis
import matplotlib.pyplot as plt
import numpy as np
from plot_set import *

from analysis.analysis_utils import *
from brokenaxes import brokenaxes

def temp(path, save = False):
    common_folder = 'temp3' 
    folders = [os.path.join(path, 'nocut', common_folder), 
               os.path.join(path, 'popup', common_folder),
               os.path.join(path, 'honeycomb', common_folder)]
    names = ['nocut', 'popup', 'honeycomb']
    # fig_max, fig_mean = variable_dependency(folders, names, 'T', '$T$ [K]', default = 300, error = 'shade')
    fig_max, fig_mean = variable_dependency(folders, names, 'T', '$T$ [K]', default = 300, error = 'shade')
    if save:
        fig_max.savefig("../article/figures/baseline/variables_temp_max_fixmove_v20.pdf", bbox_inches="tight")
        fig_mean.savefig("../article/figures/baseline/variables_temp_mean_fixmove_v20.pdf", bbox_inches="tight")

    
    
def vel(path, save = False):
    common_folder = 'vel2' 
    # common_folder = 'vel' 
    folders = [os.path.join(path, 'nocut', common_folder), 
               os.path.join(path, 'popup', common_folder),
               os.path.join(path, 'honeycomb', common_folder)]
    names = ['nocut', 'popup', 'honeycomb']
    convert = metal_to_SI(1, 's')/metal_to_SI(1,'t')
    fig_max, fig_mean = variable_dependency(folders, names, 'drag_speed', 'Sliding speed [m/s]', convert = convert, default = 20, error = 'shade')
    if save:
        fig_max.savefig("../article/figures/baseline/variables_vel_max_fixmove.pdf", bbox_inches="tight")
        fig_mean.savefig("../article/figures/baseline/variables_vel_mean_fixmove.pdf", bbox_inches="tight")

       
    
def spring(path, save = False):
    common_folder = 'spring' 
    folders = [os.path.join(path, 'nocut', common_folder), 
               os.path.join(path, 'popup', common_folder),
               os.path.join(path, 'honeycomb', common_folder)]
    names = ['nocut', 'popup', 'honeycomb']
    convert = metal_to_SI(1, 'F')/metal_to_SI(1,'s')


    fig_max = plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    ax_max = brokenaxes(xlims=((-5, 210), (240, 260)), hspace=.05)
    
    fig_mean = plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    ax_mean = brokenaxes(xlims=((-5, 210), (240, 260)), hspace=.05)
    
    bax_list = [ax_max, ax_mean]
    fig_max, fig_mean = variable_dependency(folders, names, 'K', '$K$ [N/m]', convert = convert, map = {0: 250/convert}, default = 250, figs = [[fig_max, ax_max], [fig_mean, ax_mean]])
    
    # Replace final label with 'inf'
    ax_max = fig_max.axes
    ax_mean = fig_mean.axes
    
    
    xtick_labels_max = ax_max[1].get_xticks()
    xtick_labels_mean = ax_mean[1].get_xticks()
        
    xtick_labels_max[-2] = 'inf'
    xtick_labels_mean[-2] = 'inf'
        
    ax_max[1].set_xticklabels(xtick_labels_max)
    ax_mean[1].set_xticklabels(xtick_labels_mean)
    
    # Fix moving breaklines
    for bax in bax_list:
        for handle in bax.diag_handles:
            handle.remove()
        bax.draw_diags()
    
    if save:
        fig_max.savefig("../article/figures/baseline/variables_spring_max_fixmove.pdf", bbox_inches="tight")
        fig_mean.savefig("../article/figures/baseline/variables_spring_mean_fixmove.pdf", bbox_inches="tight")

      
    
def dt(path, save = False):
    common_folder = 'dt2' 
    folders = [os.path.join(path, 'nocut', common_folder), 
               os.path.join(path, 'popup', common_folder),
               os.path.join(path, 'honeycomb', common_folder)]
    names = ['nocut', 'popup', 'honeycomb']
    convert = 1e3 # ps -> fs
    fig_max, fig_mean = variable_dependency(folders, names, 'dt', '$dt$ [fs]', convert = convert, default = 1)
    if save:
        fig_max.savefig("../article/figures/baseline/variables_dt_max_fixmove.pdf", bbox_inches="tight")
        fig_mean.savefig("../article/figures/baseline/variables_dt_mean_fixmove.pdf", bbox_inches="tight")

    
    
def variable_dependency(folders, names, variable_key, xlabel, convert = None, error = 'shade', map = None, default = None, figs = None):
    mean_window_pct = 0.5 # relative length of the mean window [% of total duration]
    std_window_pct = 0.2  # relative length of the std windoe [% of mean window]
    
    if figs is None:
        fig_max = plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
        ax_max = plt.gca()
        
        fig_mean = plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
        ax_mean = plt.gca()
        
    else:
        [fig_max, ax_max], [fig_mean, ax_mean] = figs
        
  
  
    
    markers = ['o', '^', 'D']
    colors = [color_cycle(0), color_cycle(1), color_cycle(3)]


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
            Ff_mean_std[j] = data['Ff_std'][0]*np.abs(data['Ff'][0, 1]) # rel. error -> abs. error
            
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
    
        
        color_and_label = {'color': colors[i], 'label': names[i]}
        line_and_marker = {'linestyle': '', 'marker': markers[i], 'markersize': 4}
        
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


def multi_stretch(path, save = False):
    common_folder = 'multi_stretch' 
    folders = [os.path.join(path, 'nocut', common_folder), 
               os.path.join(path, 'popup', common_folder),
               os.path.join(path, 'honeycomb', common_folder)]
    names = ['nocut', 'popup', 'honeycomb']
    
    # Mean
    vars = ['data[\'stretch_pct\']', 'data[\'Ff\'][:, :, 0, 1]', 'data[\'F_N\']']
    axis_labels = [r'Stretch', r'$\langle F_\parallel \rangle$ [nN]', r'$F_N$ [nN]']
    # yerr = 'data[\'Ff_std\'][:,:,0]*data[\'Ff\'][:,:,0, 1]'
    yerr = None
    fig_mean = multi_plot_compare(folders, names, vars, axis_labels, yerr)
    
    
    # Max
    vars = ['data[\'stretch_pct\']', 'data[\'Ff\'][:, :, 0, 0]', 'data[\'F_N\']']
    axis_labels = [r'Stretch', r'$\max \ F_\parallel$ [nN]', r'$F_N$ [nN]']
    fig_max = multi_plot_compare(folders, names, vars, axis_labels)
    
    if save:
        fig_mean.savefig("../article/figures/baseline/multi_stretch_mean_compare.pdf", bbox_inches="tight")
        fig_max.savefig("../article/figures/baseline/multi_stretch_max_compare.pdf", bbox_inches="tight")
        
        
def multi_area(path, save = False):
    common_folder = 'multi_stretch' 
    folders = [os.path.join(path, 'nocut', common_folder), 
               os.path.join(path, 'popup', common_folder),
               os.path.join(path, 'honeycomb', common_folder)]
    names = ['nocut', 'popup', 'honeycomb']
    
    # Mean
    # vars = ['data[\'stretch_pct\']', 'data[\'Ff\'][:, :, 0, 1]', 'data[\'F_N\']']
    vars = ['data[\'stretch_pct\']', 'data[\'contact_mean\'][:, :, 0]', 'data[\'F_N\']']
    axis_labels = [r'Stretch', r'Rel. $\langle$Bond$\rangle$', r'$F_N$ [nN]']
    # yerr = 'data[\'Ff_std\'][:,:,0]*data[\'Ff\'][:,:,0, 1]'
    # yerr = 'data[\'contact_std\'][:, :, 0]'
    yerr = None
    fig_mean = multi_plot_compare(folders, names, vars, axis_labels, yerr, rupplot = True)
    

    if save:
        fig_mean.savefig("../article/figures/baseline/multi_stretch_area_compare.pdf", bbox_inches="tight")
        
             
def multi_FN(path, save = False):
    common_folder = 'multi_FN' 
    folders = [os.path.join(path, 'nocut', common_folder), 
               os.path.join(path, 'popup', common_folder),
               os.path.join(path, 'honeycomb', common_folder)]
    names = ['nocut', 'popup', 'honeycomb']
    
    # Mean
    vars = ['data[\'F_N\']', 'data[\'Ff\'][:, :, 0, 1].T', 'data[\'stretch_pct\']']
    axis_labels = [r'$F_N$ [nN]', r'$\langle F_\parallel \rangle$ [nN]', r'Stretch']
    # yerr = 'data[\'Ff_std\'][:,:,0]*data[\'Ff\'][:,:,0, 1]'
    yerr = None
    fig_mean = multi_plot_compare(folders, names, vars, axis_labels, yerr, axis_scale = ['log', 'linear'], colorbar_scale = 'linear', equal_axes = [False, False], rupplot = False)
    
    # return 
    # Max
    vars = ['data[\'F_N\']', 'data[\'Ff\'][:, :, 0, 0].T', 'data[\'stretch_pct\']']
    axis_labels = [r'$F_N$ [nN]', r'$\max \ F_\parallel$ [nN]', r'Stretch']
    fig_max = multi_plot_compare(folders, names, vars, axis_labels, axis_scale = ['log', 'linear'], colorbar_scale = 'linear', equal_axes = [False, False], rupplot = False)
    
    if save:
        fig_mean.savefig("../article/figures/baseline/multi_FN_mean_compare.pdf", bbox_inches="tight")
        fig_max.savefig("../article/figures/baseline/multi_FN_max_compare.pdf", bbox_inches="tight")

    
def multi_plot_compare(folders, names, vars, axis_labels, yerr = None, axis_scale = ['linear', 'linear'], colorbar_scale = 'log', equal_axes = [False, True], rupplot = False):
    # Settings
    
    
    mean_window_pct = 0.5 # relative length of the mean window [% of total duration]
    std_window_pct = 0.2  # relative length of the std windoe [% of mean window]
    line_and_marker = {'linestyle': '', 
                       'marker': 'o',
                       'linewidth': 1.5,
                       'markersize': 2.5}
    
    cmap = matplotlib.cm.viridis
    
    
    
    grid = (1, len(folders)+1)
    width_ratios = [1 for i in range(len(folders))] + [0.1] # Small width for colorbar
    
    
    rupture_stretch = np.full((len(folders), 2), np.nan) 
    fig, axes = plt.subplots(grid[0], grid[1],  figsize = (10,5), gridspec_kw ={'width_ratios': width_ratios})
    
    # Loop through data folders
    for f, folder in enumerate(folders):
            axes[f].set_title(names[f])
            data = read_multi_folder(folder, mean_window_pct, std_window_pct)
            
            
            if False:
                print(f, folder)
                mu_mean = data['mu_mean']
                mu_max = data['mu_max']
                
                print("mu mean")
                print([f'{mu_mean[0][i]:0.{decimals(mu_mean[1][i])}f} +- {mu_mean[1][i]:1.0e}' for i in range(len(mu_mean[0])) if ~np.isnan(mu_mean[1][i])])
                print("mu max")
                print([f'{mu_max[0][i]:0.{decimals(mu_max[1][i])}f} +- {mu_max[1][i]:1.0e}' for i in range(len(mu_max[0])) if ~np.isnan(mu_max[1][i])])
            
            
            # Get variables of interest
            locs = locals()
            x, y, z = [eval(v, locs) for v in vars]
            
            # Plot
            if yerr is not None:
                f_yerr = eval(yerr)
                
            for k in range(len(z)):
                color = get_color_value(z[k], np.min(z), np.max(z), scale = colorbar_scale, cmap = cmap)
                axes[f].plot(x, y[:,k], **line_and_marker, color = color)
                
                if yerr is not None:
                    # xlim, ylim = axes[f].get_xlim(), axes[f].get_ylim()
                    # axes[f].errorbar(x, y[:,k], yerr = f_yerr[:,k], **line_and_marker, color = color, capsize=6) 
                    axes[f].fill_between(x, y[:,k] + f_yerr[:,k], y[:,k] - f_yerr[:,k], alpha = 0.1, color = color)
                    # axes[f].set_xlim(xlim); axes[f].set_ylim(ylim)              
  
                axes[f].set_xscale(axis_scale[0])
                axes[f].set_yscale(axis_scale[1])
                    
                    
            # Get rupture strecth information
            rupture_stretch[f] = (data['rupture_stretch'], data['practical_rupture_stretch'])
           
                
    
    # Colorbar
    if colorbar_scale == 'linear':
        norm = matplotlib.colors.BoundaryNorm(z, cmap.N)
    elif colorbar_scale == 'log':
        norm = matplotlib.colors.LogNorm(z[0], z[-1])
    else:
        exit(f'scale = \'{colorbar_scale}\' is not defined.')
        
    axes[-1].grid(False)
    axes[-1].set_aspect(10)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axes[-1])
    cb.set_label(label = axis_labels[2], fontsize=14)
    
    # Axis limits
    if np.any(equal_axes):
        xlim = [ax.get_xlim() for ax in axes[:-1]]
        ylim = [ax.get_ylim() for ax in axes[:-1]]
        
        xlim = [np.min(xlim), np.max(xlim)]
        ylim = [np.min(ylim), np.max(ylim)]
        for ax in axes[:-1]:
            if equal_axes[0]: ax.set_xlim(xlim)
            if equal_axes[1]: ax.set_ylim(ylim)
     
    
    # Rupture stretch 
    if rupplot: 
        for a, ax in enumerate(axes[:-1]):    
            vline(ax, rupture_stretch[a, 0], linestyle = '--', color = 'black', linewidth = 1, zorder = 0, label = "Rupture stretch" )
            yfill(ax, [rupture_stretch[a, 1], 10], color = 'red', alpha = 0.1, zorder = 0, label = "Rupture drag")


    # Axis scale 
    
         
    # labels and legends
    fig.supxlabel(axis_labels[0], fontsize = 14)
    fig.supylabel(axis_labels[1], fontsize = 14)
    handles, labels = axes[-2].get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'lower right', bbox_to_anchor = (0.0, 0.0, 1, 1), bbox_transform = plt.gcf().transFigure, ncols = 2, fontsize = 13)
    
    
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    return fig        


def contact_vs_time(path, save = False):
    bond_file = 'bond_pct.txt'
    info_file = 'info_file.txt'
    
    dirs = ['nocut/contact/nocut_contact',  
            'popup/contact/pop_contact', 
            'honeycomb/contact/hon_contact']
         
    names = ['nocut', 'popup', 'honeycomb']
    
    colors = [color_cycle(0), color_cycle(1), color_cycle(3)]
    
    
    
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    for i, dir in enumerate(dirs):
            
        timestep, contact_full_sheet, contact_inner_sheet = np.loadtxt(os.path.join(path, dir, bond_file), unpack=True)
        info = read_info_file(os.path.join(path, dir, info_file))
        time = timestep * info['dt']
        stretch_start = time >= info['relax_time']
        
        contact = contact_full_sheet[stretch_start]
        stretch = (time[stretch_start] - time[stretch_start][0]) * info['stretch_speed_pct']
        
        
        if info['is_ruptured']:
            # plt.plot(stretch[-1], contact[-1], 'o', color = color_cycle(i))
            rup = info['rupture_stretch']
            vline(plt.gca(), rup, linestyle = "--", alpha = 1, linewidth = 1, color = color_cycle(i))
            names[i] += f', rupture = {rup:0.2f}'


        
        plt.plot(stretch, contact, color = colors[i], label = names[i])
        plt.xlabel('Stretch', fontsize=14)
        plt.ylabel('Rel. Bond', fontsize=14)
        plt.legend(fontsize = 13)
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        if save:
            plt.savefig("../article/figures/baseline/contact_vs_stretch", bbox_inches="tight")


def vaccum_normal_buckling(path, save = False):
    info_file = 'info_file.txt'
    
    
    dump_files = ['full_sheet_nocut_vacuum.data',
                 'full_sheet_pop_vacuum.data',
                 'full_sheet_hon_vacuum.data']
    
    
    dirs = ['nocut/vacuum/nocut_vacuum',
            'popup/vacuum/pop_vacuum',
            'honeycomb/vacuum/hon_vacuum']
    
    names = ['nocut', 'popup', 'honeycomb']
    
    colors = [color_cycle(0), color_cycle(1), color_cycle(3)]
    
    

    
    # dump_files.pop(0)
    # dirs.pop(0)
    # names.pop(0)
    # colors.pop(0)
    
    grid = (1, len(dump_files))
    fig, axes = plt.subplots(grid[0], grid[1],  figsize = (10,5))
    

    # plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    for i, dir in enumerate(dirs):
        info = read_info_file(os.path.join(path, dir, info_file))
        timestep, Q_var, Q = get_normal_buckling(os.path.join(path, dir, dump_files[i]), quartiles = [0.01, 0.1, 0.25, 0.50])

        print(f'Rupture stretch = {info["rupture_stretch"]}')
        time = timestep * info['dt']
        stretch_start = time >= info['relax_time']
        
        Q = Q[:, stretch_start]
        stretch = (time[stretch_start] - time[stretch_start][0]) * info['stretch_speed_pct']


        axes[i].set_title(names[i])
        for j in range(Q.shape[0]//2):
            label = f"Q = {Q_var[-j-1]*100:2.0f}%, {Q_var[j]*100:2.0f}%"
            if Q_var[j] == 1:
                label = 'Q = Min, Max'
            axes[i].plot(stretch, Q[j], color = color_cycle(j), label = label)
            axes[i].plot(stretch, Q[-j-1], color = color_cycle(j))
        
        if Q.shape[0]%2 == 1:
            j += 1
            axes[i].plot(stretch, Q[j], color = color_cycle(j), label = f"Q = Median")
            
            
        
        # plt.plot(, contact, color = colors[i], label = names[i])
        # plt.xlabel('Stretch', fontsize=14)
        # plt.ylabel('Rel. Bond', fontsize=14)
        # plt.legend(fontsize = 13)
        # plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)

    fig.supxlabel('Stretch', fontsize = 14)
    fig.supylabel('z-position [Å]', fontsize = 14)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'right', bbox_to_anchor = (0.0, 0.0, 1, 1), bbox_transform = plt.gcf().transFigure, ncols = 1, fontsize = 13)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.subplots_adjust(right=0.8)
    if save:
        plt.savefig("../article/figures/baseline/vacuum_normal_buckling", bbox_inches="tight")

if __name__ == "__main__":
    
    # path = '../Data/Baseline'
    # temp(path, save = False)
    
    path = '../Data/Baseline_fixmove'
    # path = '../Data/Baseline'
    # temp(path, save = False)
    # vel(path, save = False)
    # spring(path, save = False)
    # dt(path, save = False)
    
    multi_stretch(path, save = False)
    # multi_FN(path, save = False)
    # multi_area(path, save = False)
    
    # contact_vs_time(path, save = False)
    # vaccum_normal_buckling(path, save = False)
    
    
    plt.show()