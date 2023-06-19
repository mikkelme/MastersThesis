
import sys
sys.path.append('../') # parent folder: MastersThesis
import matplotlib.pyplot as plt
import numpy as np
from plot_set import *
from analysis.analysis_utils import *
from produce_figures.negative_coef import *

import matplotlib.legend_handler

def force_traces(save = False):
    # File
    path = '../Data/Baseline_fixmove' 
    filename = os.path.join(path,'nocut/temp/T300/system_T300_Ff.txt') 
    
    
    
    
    # Data
    info, data = analyse_friction_file(filename, mean_window_pct = 0.5, std_window_pct = 0.35)    
    time = data['time'] - data['time'][0]
    VA_pos = time * info['drag_speed']  # virtual atom position
    Ff = data[f'Ff_full_sheet'][:,0]
 
 
    window_length = 150; polyorder = 5
    Ff_savgol = savgol_filter(window_length, polyorder, Ff)[0]
    print(f'window length = {window_length}, corresponding to drag distance {VA_pos[window_length]} Å and time {time[window_length]} ps')
 
 
    # --- Figure 1 --- #
    # (VA_pos, Ff full sheet parallel) | drag length = 10 Å
    map = [VA_pos <= 100][0]
    fig1 = plt.figure(num = unique_fignum(), dpi=100, facecolor='w', edgecolor='k'); ax1 = fig1.gca()
    ax1.set_facecolor('white')
    plt.plot(VA_pos[map], Ff[map], label = "Raw data")
    plt.plot(VA_pos[map], Ff_savgol[map], label = f"Savgol filter")
    plt.xlabel(r'Sliding distance (Å)', fontsize=20)
    plt.ylabel(r'Friction force $F$ (nN)', fontsize=20)
    plt.legend(loc = 'lower left', fontsize = 20, ncol = 2)
    add_xaxis(plt.gca(), x = VA_pos[map], xnew = time[map], xlabel = 'Time (ps)', decimals = 0, fontsize = 20)
    
    plt.ylim(top = 2.5)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    # --- Figure 1 --- #
    # (VA_pos, Ff full sheet parallel) | drag length = 100 Å
    map = [VA_pos <= 10][0]
    fig2 = plt.figure(num = unique_fignum(), dpi=100, facecolor='w', edgecolor='k'); ax2 = fig2.gca()
    ax1.set_facecolor('white')
    
    plt.plot(VA_pos[map], Ff[map], label = "Raw data")
    plt.plot(VA_pos[map], Ff_savgol[map], label = f"Savgol filter")
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    add_xaxis(plt.gca(), x = VA_pos[map], xnew = time[map], xlabel = '', decimals = 0, fontsize = 14)
    plt.xticks(fontsize=40)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    
    
    
    if save:
        fig1.savefig('../article/figures/fig1_Ff_100Å.pdf', bbox_inches='tight')
        fig2.savefig('../article/figures/fig1_drag_Ff_10Å.pdf', bbox_inches='tight')
        
    
def multi_plot(save = False):
    # Files
    path = '../Data/Baseline_fixmove'
    common_folder = 'multi_stretch' 
    folders = [os.path.join(path, 'nocut', common_folder), 
               os.path.join(path, 'popup', common_folder),
               os.path.join(path, 'honeycomb', common_folder)]
    
    
    

    
    # Settings
    mean_window_pct = 0.5 # relative length of the mean window [% of total duration]
    std_window_pct = 0.35  # relative length of the std windoe [% of mean window]
    line_and_marker = {'marker': 'o',
                       's': 20,
                       'edgecolor': 'black'}
    names = ['No cut', 'Tetrahedron', 'Honeycomb']
    cmap = matplotlib.cm.viridis
    
    cm_to_inch = 1/2.54
    # fontsize = 0.2*cm_to_inch*72
    fontsize = 10
    
    # Figure    
    fig, axes = plt.subplots(2, 3, num = unique_fignum(), figsize = (2*8.6*cm_to_inch,10*cm_to_inch))
    
    # Arrays
    rupture_stretch = np.full((3, 2), np.nan) 
    
    
    
    # Loop through data folders
    for f, folder in enumerate(folders):
            axes[0, f].set_title(names[f], fontsize = fontsize)
            data = read_multi_folder(folder, mean_window_pct, std_window_pct)
            strain = data['stretch_pct']
            F = data['Ff'][:, :, 0, 1]
            contact = data['contact_mean'][:, :, 0]
            F_N = data['F_N']
            
            for k in range(len(F_N)):     
                color = get_color_value(F_N[k], 0.1, 10, scale = 'log', cmap = cmap)
                axes[0, f].scatter(strain, contact[:,k], **line_and_marker, color = color)
                axes[1, f].scatter(strain, F[:,k], **line_and_marker, color = color)

                                         
                    
                    
            # Get rupture strecth information
            rupture_stretch[f] = (data['rupture_stretch'], data['practical_rupture_stretch'])
           

    # Marker labels
    for k in range(3):
        color = get_color_value(F_N[k], 0.1, 10, scale = 'log', cmap = cmap)
        axes[0, 0].scatter([], [], **line_and_marker, color = color, label = f'{F_N[k]} nN')


    
    for i in range(2):
        # Axis limits
        ylim = [ax.get_ylim() for ax in axes[i]]
        ylim = [np.min(ylim), np.max(ylim)]
        for ax in axes[i]:
            ax.set_ylim(ylim)
     
            # Set tick size
            ax.tick_params(axis='both',  labelsize = fontsize)
            ax.set_axisbelow(True)
            
            # More gridlines
            # ax.minorticks_on()
            # ax.tick_params(axis='both',  labelsize = fontsize)
            # ax.grid(True, which = 'both') 
            # ax.set_axisbelow(True)
     
     
   

        # Rupture stretch 
        for a in range(len(rupture_stretch)):
            vline(axes[i, a], rupture_stretch[a, 0], linestyle = '--', color = 'black', linewidth = 1, zorder = 0, label = "Rupture test" )
            yfill(axes[i, a], [rupture_stretch[a, 1], 10], color = 'red', alpha = 0.1, zorder = 0, label = "Rupture sliding")


    
    # labels and legends
    fig.supxlabel('Strain', fontsize = fontsize)
    axes[0,0].legend(loc = 'lower left', fontsize = fontsize)    
    axes[0,0].set_ylabel(r'$\langle$ Rel. Contact $\rangle$', fontsize = fontsize)
    axes[1,0].set_ylabel(r'$\langle F \rangle$ (nN)', fontsize = fontsize)    
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)


    if save:
        fig.savefig("../article/figures/fig2.pdf", bbox_inches="tight")
       
 
 
 
def coupled_system(path, compare_path = None, save = False, add_path = None, add_stretch_range = None, markings = None):
    
    cm_to_inch = 1/2.54
    fontsize = 5
    
    
    # Settings
    mean_window_pct = 0.5 # relative length of the mean window [% of total duration]
    std_window_pct = 0.35  # relative length of the std windoe [% of mean window]
    cmap = matplotlib.cm.viridis
    stretch_tension_file = 'stretch_tension.txt' 
    
    

    color_coupled = color_cycle(1)
    color_added = color_cycle(4)
    
    marker_coupled = 'v'
    marker_added = '^'
    marker_non_coupled = 'o'
    
    size_and_edge = {'s': 10, 'edgecolors': "black"}
    markings_settings = {'marker': 's', 's': 20, 'alpha': 0.8, 'edgecolors': "black", 'color': 'tab:pink', 'zorder': 20}
    
    
    # Plot 
    fig, axes = plt.subplots(1, 2, num = unique_fignum(), figsize = (8.6*cm_to_inch,1/2*8.6*cm_to_inch))
    
    for ax in axes:
        # Set tick size
        ax.tick_params(axis='both',  labelsize = fontsize)
        ax.set_axisbelow(True)
    
    

    
    
  
    # --- Data --- #
    # Get load (tension) vs stretch
    stretch_tension = read_friction_file(os.path.join(path, stretch_tension_file))
    rupture_dict = read_info_file(os.path.join(path, 'rupture_test.txt'))
    stretch_test = stretch_tension['v_stretch_pct']
    load_test = metal_to_SI(stretch_tension['v_load'], 'F')*1e9
    tension_test = metal_to_SI(stretch_tension['v_tension'], 'F')*1e9
    
    # Coupling data
    data = read_multi_coupling(path, mean_window_pct, std_window_pct)
    info = read_info_file(os.path.join(path, 'info_file.txt'))
    stretch = data['mean_stretch']
    stretch_initial = data['stretch_pct']
    std_stretch = data['std_stretch']
    non_rup = data['rup'] < 1
    
    
    # Add data coupling
    if add_path is not None:
        add_data = read_multi_coupling(add_path, mean_window_pct, std_window_pct)
        add_info = read_info_file(os.path.join(add_path, 'info_file.txt'))
        add_stretch = add_data['mean_stretch']
        add_stretch_initial = add_data['stretch_pct']
        add_std_stretch = add_data['std_stretch']
        add_non_rup = add_data['rup'] < 1
        add_F_N = add_data['F_N'] # Full sheet
        add_Ff = add_data['Ff'][:, 0, 1]
        
        if add_stretch_range is None:
            add_stretch_map =  ~np.isnan(add_stretch[add_non_rup])
        else: 
            add_stretch_map = np.logical_and(add_stretch_range[0] <= add_stretch[add_non_rup], add_stretch[add_non_rup] <= add_stretch_range[1])
        
            
    F_N = data['F_N'] # Full sheet
    Ff = data['Ff'][:, 0, 1]
    vmin = 0.1
    vmax = 10
    
    # Compare data (without coupling)
    if compare_path is not None:
        data = read_multi_folder(compare_path, mean_window_pct, std_window_pct)    
        stretch_compare = data['stretch_pct']
        Ff_compare = data['Ff'][:, :, 0, 1]
        F_N_compare = data['F_N']
        
    
    
    # --- Plotting --- #
    ## Stretch vs. normal force (tension) (left plot) ##
    # Original coupling
    axes[0].scatter(F_N[non_rup], stretch[non_rup], marker = marker_coupled, **size_and_edge,  color = color_coupled) # Mean stretch 
    # axes[0].scatter(F_N[non_rup], stretch_initial[non_rup], marker = marker_initial_stretch, **size_and_edge, color = color_original, alpha = 0.5) # Initial stretch 
    
    F_N_concat = F_N[non_rup]
    stretch_concat = stretch[non_rup]
    Ff_concat = Ff[non_rup]
    
    # Add data coupling
    if add_path is not None:
        axes[0].scatter(add_F_N[add_non_rup][add_stretch_map], add_stretch[add_non_rup][add_stretch_map], marker = marker_added, **size_and_edge,  color = color_added) # Mean stretch
        # axes[0].scatter(add_F_N[add_non_rup][add_stretch_map], add_stretch_initial[add_non_rup][add_stretch_map], marker = marker_initial_stretch, **size_and_edge, color = color_added, alpha = 0.5) # Initial stretch
    
        A_idx = stretch_concat < add_stretch_range[0]
        B_idx = stretch_concat > add_stretch_range[1]
        F_N_concat = np.concatenate((F_N_concat[A_idx], add_F_N[add_non_rup][add_stretch_map] , F_N_concat[B_idx]))
        stretch_concat = np.concatenate((stretch_concat[A_idx], add_stretch[add_non_rup][add_stretch_map] , stretch_concat[B_idx]))
        Ff_concat = np.concatenate((Ff_concat[A_idx], add_Ff[add_non_rup][add_stretch_map] , Ff_concat[B_idx]))
    
    
    # Interpolation for strain -> load mapping 
    strain_to_load = interpolate.interp1d(stretch_concat, F_N_concat)
    
    
    new_ax = add_xaxis(axes[0], x = load_test, xnew = load_test*rupture_dict['R'], xlabel = 'Tension [nN]', decimals = 1, fontsize = fontsize)
    new_ax.tick_params(axis='both',  labelsize = fontsize)
    new_ax.set_axisbelow(True)

    
    axes[0].set_xlabel(r'$F_N$ [nN]', fontsize = fontsize)
    axes[0].set_ylabel('Strain', fontsize = fontsize)

   
    ## Friction vs. normal force (right plot) ##
    # Original coupling
    axes[1].scatter(strain_to_load(stretch[non_rup]), Ff[non_rup], marker = marker_coupled, **size_and_edge, color = color_coupled, zorder = 10)
    
    # Add data coupling
    if add_path is not None:
        axes[1].scatter(strain_to_load(add_stretch[add_non_rup][add_stretch_map]), add_Ff[add_non_rup][add_stretch_map], marker = marker_added, **size_and_edge, color = color_added, zorder = 10)
        
        
    if markings: # Add markings add certain strains
        smin, smax = np.min(stretch_concat), np.max(stretch_concat)
        for i, mark in enumerate(markings):
            if smin < mark and mark < smax:
                idx = np.argmin(np.abs(stretch_concat - mark))
                frame = strain_to_frame(stretch_concat[idx])
                print(f'mark {i}: strain = {stretch_concat[idx]}, F_N = {F_N_concat[idx]}, ovito frame = {frame}')
                axes[0].scatter(F_N_concat[idx], stretch_concat[idx], **markings_settings) 
                axes[1].scatter(F_N_concat[idx], Ff_concat[idx], **markings_settings)
                                                
            else:
                print(f'mark {mark} not in range [{smin}, {smax}]') 
        
                        
        
         
 
    # Compare non-coupled
    if compare_path is not None:     
        for k in range(len(F_N_compare)):
                color = get_color_value(F_N_compare[k], vmin, vmax, scale = 'log', cmap = cmap)
                valid = np.logical_and(~np.isnan(Ff_compare[:, k]), stretch_compare <= stretch_concat[-1])
                axes[1].scatter(strain_to_load(stretch_compare[valid]), Ff_compare[valid,k], marker = marker_non_coupled, **size_and_edge, color = color)
                

    axes[1].set_xlabel(r'Strain $\to$ $F_N$ [nN]', fontsize = fontsize)
    axes[1].set_ylabel(r'$\langle F_\parallel \rangle$ [nN]', fontsize = fontsize)
    
    
    # --- Add legends --- #    
    if add_path:
        L1 = axes[0].scatter([], [], color = color_coupled, **size_and_edge, marker = marker_coupled, label = 'Coupled (sim 1, sim2)')
        L2 = axes[0].scatter([], [], color = color_added, **size_and_edge, marker = marker_added)
    else:
        L1 = axes[0].scatter([], [], color = color_coupled, **size_and_edge, marker = marker_coupled, label = 'Coupled')

    L31 = axes[0].scatter([], [], color = get_color_value(0.1, vmin, vmax, scale = 'log', cmap = cmap), **size_and_edge, marker = marker_non_coupled, label = 'Non-coupled\n(0.1, 1, 10 nN)')
    L32 = axes[0].scatter([], [], color = get_color_value(1, vmin, vmax, scale = 'log', cmap = cmap), **size_and_edge, marker = marker_non_coupled)
    L33 = axes[0].scatter([], [], color = get_color_value(10, vmin, vmax, scale = 'log', cmap = cmap), **size_and_edge, marker = marker_non_coupled)
    
    
    if add_path:
        handles = [(L1, L2), (L31, L32, L33)]
    else:
        handles = [L1, (L31, L32, L33)]
    _, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles = handles, labels=labels, fontsize = fontsize, loc = 'lower right', handler_map = {tuple: matplotlib.legend_handler.HandlerTuple(None)})

    
    markings_settings['alpha'] = 1
    L4 = axes[1].scatter([], [], **markings_settings, label = 'Frames')
    axes[1].legend(fontsize = fontsize, loc = 'lower right')
    
    


   
    # Wrap it up
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)    
    if save is not False:
        plt.savefig(f'../article/figures/{save}.pdf', bbox_inches='tight')

      
       
def plot_coupled_system(save = False):
    path = '../Data/negative_coef/multi_coupling_free_popup'
    compare_path = '../Data/Baseline_fixmove/popup/multi_stretch'
    if save is not False:
        save = 'fig3a'
    markings = [0.04, 0.08, 0.12, 0.165] # strain 
    coupled_system(path, compare_path, save, markings = markings)
        
    
    # path = '../Data/negative_coef/multi_coupling_free_honeycomb'
    # add_path = '../Data/negative_coef/multi_coupling_free_honeycomb_zoom'
    # add_stretch_range = [0.1, 0.65]
    # compare_path = '../Data/Baseline_fixmove/honeycomb/multi_stretch'
    # if save is not False:
    #     save = 'fig3b'
    # markings = [0.065, 0.76, 1.04, 1.15] # strain  #0.6448
    # coupled_system(path, compare_path, save, add_path, add_stretch_range, markings = markings)
      


if __name__ == '__main__':
    # force_traces(save = True)
    # multi_plot(save = True)
    plot_coupled_system(save = True)
    plt.show()