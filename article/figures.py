
import sys
sys.path.append('../') # parent folder: MastersThesis
import matplotlib.pyplot as plt
import numpy as np
from plot_set import *
from analysis.analysis_utils import *

# from scipy.signal import argrelextrema
# from brokenaxes import brokenaxes




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
    plt.xlabel(r'Sliding distance [Å]', fontsize=20)
    plt.ylabel(r'Friction force $F_\parallel$ [nN]', fontsize=20)
    plt.legend(loc = 'lower left', fontsize = 20, ncol = 2)
    add_xaxis(plt.gca(), x = VA_pos[map], xnew = time[map], xlabel = 'Time [ps]', decimals = 0, fontsize = 20)
    
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
    fontsize = 13
    # Figure    
    cm_to_inch = 1/2.54
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
  
  
  
            # axes[f].set_xscale(axis_scale[0])
            # axes[f].set_yscale(axis_scale[1])
            # for a in range(len(axis_scale)):
            #     # print(f'{a} | {axis_scale[a]} == log = {axis_scale[a] == "log"}')
            #     if axis_scale[a] == 'log':
            #         if a == 0:
            #             ax = axes[f].xaxis
            #         elif a == 1:
            #             ax = axes[f].yaxis
                        
            #         ax.get_major_locator().set_params(numticks=99)
            #         ax.get_minor_locator().set_params(numticks=99, subs=[.2, .4, .6, .8])
                    
            #         # axes[f].tick_params(axis='both')
            #         # axes[f].grid(True, which = 'both') 
                        
                         
                
                    
                    
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
     

        # Rupture stretch 
        for a in range(len(rupture_stretch)):
            vline(axes[i, a], rupture_stretch[a, 0], linestyle = '--', color = 'black', linewidth = 1, zorder = 0, label = "Rupture test" )
            yfill(axes[i, a], [rupture_stretch[a, 1], 10], color = 'red', alpha = 0.1, zorder = 0, label = "Rupture sliding")


    
    # labels and legends
    fig.supxlabel('Strain', fontsize = fontsize)
    axes[0,0].legend(loc = 'lower left', fontsize = 10)    
    axes[0,0].set_ylabel(r'$\langle$ Rel. Contact $\rangle$', fontsize = fontsize)
    axes[1,0].set_ylabel(r'$\langle F_\parallel \rangle$ [nN]', fontsize = fontsize)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)


    if save:
        fig.savefig("../article/figures/fig2.pdf", bbox_inches="tight")
        


if __name__ == '__main__':
    # force_traces(save = False)
    multi_plot(save = True)
    plt.show()