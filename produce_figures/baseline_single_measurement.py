import sys
sys.path.append('../') # parent folder: MastersThesis
import matplotlib.pyplot as plt
import numpy as np
from plot_set import *
from analysis.analysis_utils import *

from scipy.signal import argrelextrema


def raw_data(filename, save = False):
    """ Raw data """
    
    mean_window_pct = 0.5 # relative length of the mean window [% of total duration]
    std_window_pct = 0.35  # relative length of the std windoe [% of mean window]
    
    
    info, data = analyse_friction_file(filename, mean_window_pct, std_window_pct)    
    time = data['time'] - data['time'][0]
    VA_pos = time * info['drag_speed']  # virtual atom position
    Ff = data[f'Ff_full_sheet'][:,0]
 
    # --- Figure 1 --- #
    # (VA_pos, Ff full sheet parallel) | drag length = 10 Å
    map = [VA_pos <= 10][0]
    window_length = 150; polyorder = 5
    print(f'window length = {window_length}, corresponding to drag distance {VA_pos[window_length]} Å and time {time[window_length]} ps')
    Ff_savgol = savgol_filter(window_length, polyorder, Ff)[0]
    
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(VA_pos[map], Ff[map], label = "Raw data")
    plt.plot(VA_pos[map], Ff_savgol[map], label = f"Savgol filter")
    # plt.plot(VA_pos[map], data[f'move_force'][:,0][map], label = f"Moving body force") # XXX
    
    plt.xlabel(r'Sliding distance [Å]', fontsize=14)
    plt.ylabel(r'Friction force $F_\parallel$ [nN]', fontsize=14)
    plt.legend(loc = 'lower left', fontsize = 13)
    
    add_xaxis(plt.gca(), x = VA_pos[map], xnew = time[map], xlabel = 'Time [ps]', decimals = 0, fontsize = 14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
        # plt.savefig('../article/figures/baseline/drag_Ff_10Å.pdf', bbox_inches='tight')
        # plt.savefig('../article/figures/baseline/drag_Ff_10Å_K10_v1.pdf', bbox_inches='tight')
        pass
    
    
    # --- Figure 2 --- #
    # (VA_pos, Ff full sheet parallel) | drag length = 50 Å
    map = [VA_pos <= 100][0]
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(VA_pos[map], Ff[map], label = "Raw data")
    plt.plot(VA_pos[map], Ff_savgol[map], label = f"Savgol filter")
    plt.xlabel(r'Sliding distance [Å]', fontsize=14)
    plt.ylabel(r'Friction force $F_\parallel$ [nN]', fontsize=14)
    plt.legend(loc = 'lower left', fontsize = 13)
    
    add_xaxis(plt.gca(), x = VA_pos[map], xnew = time[map], xlabel = 'Time [ps]', decimals = 0, fontsize = 14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
        # plt.savefig('../article/figures/baseline/drag_Ff_100Å.pdf', bbox_inches='tight')
        # plt.savefig('../article/figures/baseline/drag_Ff_100Å_K30_v1.pdf', bbox_inches='tight')
        pass
    



def ft(filename, save = False):
    """ Fourier transform """
    mean_window_pct = 0.5 # relative length of the mean window [% of total duration]
    std_window_pct = 0.35  # relative length of the std windoe [% of mean window]
    
    info, data = analyse_friction_file(filename, mean_window_pct, std_window_pct)    
    time = data['time'] - data['time'][0]
    VA_pos = time * info['drag_speed']  # virtual atom position
    Ff = data[f'Ff_full_sheet'][:,0]
    
    # Fourier transform
    half_range = range(int(len(Ff)/2))
    Ff_fourier = np.fft.fft(Ff)/len(Ff)  # Normalize
    Ff_fourier = Ff_fourier[half_range] # Exclude sampling frequency
    
    freq = half_range/time[-1]
    amplitude = abs(Ff_fourier)
    phase = np.imag(np.log(Ff_fourier))
    
    
    # --- figure 1 --- #
    # Full frequency range
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(freq, amplitude)
    plt.xlabel(r'Frequency [ps$^{-1}$]', fontsize=14)
    plt.ylabel(r'Amplitude', fontsize=14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
        plt.savefig('../article/figures/baseline/ft.pdf', bbox_inches='tight')
    
    # --- figure 2 --- #
    # Reduced frequency range with annotated peaks
    zoom = len(freq)//15
    peaks = [147, 158] 
    
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(freq[:zoom], amplitude[:zoom])
    plt.xlabel(r'Frequency [ps$^{-1}$]', fontsize=14)
    plt.ylabel(r'Amplitude', fontsize=14)
    
    
    for idx in peaks:
        plt.plot(freq[idx], amplitude[idx], 'ko')
        plt.text(freq[idx] - 0.045, amplitude[idx], f'$({freq[idx]:.3f})$', fontsize = 14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
        plt.savefig('../article/figures/baseline/ft_zoom.pdf', bbox_inches='tight')
    

    # --- figure 3 --- #
    # Plot dominating frequencies on top of raw data
    map = [VA_pos <= 100][0]
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(VA_pos[map], Ff[map], label = "Raw data")
    yf = 1/2*(np.sin(2*np.pi*freq[peaks[0]]*time[map] + phase[peaks[0]]) + np.sin(2*np.pi*freq[peaks[1]]*time[map] + phase[peaks[1]]))
    plt.plot(VA_pos[map], yf, label = rf'$ F_{{\parallel}} \propto \sin(2\pi \ {freq[peaks[0]]:.3f} $ ps$^{{-1}}) +  \sin(2\pi  \ {freq[peaks[1]]:.3f} $ ps$^{{-1}}) $')
    
    a = (freq[peaks[0]] + freq[peaks[1]])/2
    b = (freq[peaks[1]] - freq[peaks[0]])/2
    print(f'Wavepacket freq.: a = {a}, b = {b}, +- {freq[1]-freq[0]}')
    # yf = np.sin(2*np.pi*a*VA_pos[map])*np.cos(2*np.pi*b*VA_pos[map])
    # plt.plot(VA_pos[map], yf)
    
    plt.xlabel(r'Sliding distance [Å]', fontsize=14)
    # plt.xlabel(r'Time [ps]', fontsize=14)
    plt.ylabel(r'Friction force $F_\parallel$ [nN]', fontsize=14)
    plt.legend(loc = 'lower left', fontsize = 13)
    
    add_xaxis(plt.gca(), x = VA_pos[map], xnew = time[map], xlabel = 'Time [ps]', decimals = 0, fontsize = 14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
        plt.savefig('../article/figures/baseline/ft_sine.pdf', bbox_inches='tight')
    
    
def decomp(filename, save = False):
    mean_window_pct = 0.5 # relative length of the mean window [% of total duration]
    std_window_pct = 0.35  # relative length of the std windoe [% of mean window]
    
    info, data = analyse_friction_file(filename, mean_window_pct, std_window_pct)    
    time = data['time'] - data['time'][0]
    VA_pos = time * info['drag_speed']  # virtual atom position
    window_length = 150; polyorder = 5
    
    # --- Data --- #
    # Full para
    Ff = data[f'Ff_full_sheet'][:,0]
    
    # Full Perp
    Ff_perp = data[f'Ff_full_sheet'][:,1]
    
    # Sheet and PB para
    Ff_sheet =  data[f'Ff_sheet'][:,1]
    Ff_PB =  data[f'Ff_PB'][:,1]
    
    # Savgol
    Ff_savgol, Ff_perp_savgol, Ff_sheet_savgol, Ff_PB_savgol = savgol_filter(window_length, polyorder, Ff, Ff_perp, Ff_sheet, Ff_PB)
    
    # Map
    map = [VA_pos <= 100][0]
    
    # --- Figure 1 --- #
    # Force decomposition: Full sheet = sheet + PB
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(VA_pos[map], Ff_sheet_savgol[map], label = "Sheet")
    plt.plot(VA_pos[map], Ff_PB_savgol[map], label = "PB")
    
    plt.xlabel(r'Sliding distance [Å]', fontsize=14)
    plt.ylabel(r'Friction force $F_\parallel$ [nN]', fontsize=14)
    plt.legend(loc = 'lower left', fontsize = 13)
    
    add_xaxis(plt.gca(), x = VA_pos[map], xnew = time[map], xlabel = 'Time [ps]', decimals = 0, fontsize = 14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
         plt.savefig('../article/figures/baseline/decomp_group.pdf', bbox_inches='tight')
   
    
    # --- Figure 1 --- #
    # Force decomposition: Parallel and perpendicular
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(VA_pos[map], Ff_savgol[map], label = "$F_\parallel$", color = color_cycle(0))
    plt.plot(VA_pos[map], Ff_perp_savgol[map], label = "$F_\perp$", color = color_cycle(1))
        
    plt.xlabel(r'Sliding distance [Å]', fontsize=14)
    plt.ylabel(r'Friction force $F_\parallel$ [nN]', fontsize=14)
    plt.legend(loc = 'lower left', fontsize = 13)
    
    add_xaxis(plt.gca(), x = VA_pos[map], xnew = time[map], xlabel = 'Time [ps]', decimals = 0, fontsize = 14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    if save:
        plt.savefig('../article/figures/baseline/decomp_direc.pdf', bbox_inches='tight')
        

    
    # # Friction and move force 
    # move = data['move_force'][:, 0]
    # move_perp = data['move_force'][:, 0]
    # move_savgol, move_perp_savgol = savgol_filter(window_length, polyorder, move, move_perp)
    # Ff_perp = data[f'Ff_full_sheet'][:,1]
    # Ff_perp_savgol = savgol_filter(window_length, polyorder, Ff_perp)[0]

    # plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    # # plt.plot(VA_pos[map], Ff[map], label = "Ff")
    # # plt.plot(VA_pos[map], move[map], label = "move force")
    # plt.plot(VA_pos[map], Ff_savgol[map], label = "Ff")
    # plt.plot(VA_pos[map], move_savgol[map], label = "move force")
    # plt.xlabel(r'Sliding distance [Å]', fontsize=14)
    # plt.ylabel(r'---', fontsize=14)
    # plt.legend(fontsize = 13)
    # plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    

    # # plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    # # plt.plot(VA_pos[map], Ff_perp_savgol[map], label = "Ff perp")
    # # plt.plot(VA_pos[map], move_perp_savgol[map], label = "move perp force")
    # # plt.xlabel(r'Sliding distance [Å]', fontsize=14)
    # # plt.ylabel(r'---', fontsize=14)
    # # plt.legend(fontsize = 13)
    # # plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    

def COM(filename, save = False):
    mean_window_pct = 0.5 # relative length of the mean window [% of total duration]
    std_window_pct = 0.35  # relative length of the std windoe [% of mean window]

    info, data = analyse_friction_file(filename, mean_window_pct, std_window_pct)    
    time = data['time'] - data['time'][0]
    VA_pos = time * info['drag_speed']  # virtual atom position
    # window_length = 150; polyorder = 5
    window_length = 20; polyorder = 5

    # --- Data --- #
    # Full para
    Ff = data[f'Ff_full_sheet'][:,0]

    # Full Perp
    Ff_perp = data[f'Ff_full_sheet'][:,1]

    # Sheet and PB para
    Ff_sheet =  data[f'Ff_sheet'][:,1]
    Ff_PB =  data[f'Ff_PB'][:,1]

    # Savgol
    Ff_savgol, Ff_perp_savgol, Ff_sheet_savgol, Ff_PB_savgol = savgol_filter(window_length, polyorder, Ff, Ff_perp, Ff_sheet, Ff_PB)

    # Map
    
    # --- Figure 10 --- #
    # COM oath
    
    map = [VA_pos <= 2][0]
    # fig = plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    fig = plt.figure(num = unique_fignum(), figsize = (10,3), dpi = 80, facecolor='w', edgecolor='k')
    ax = plt.gca()
    
    speed = np.full(len(data['time']), np.nan)
    speed[:-1] = np.linalg.norm(data['COM_sheet'][1:,0:2]-data['COM_sheet'][:-1,0:2], axis = 1)/(data['time'][1]-data['time'][0])
    
    
    plot_xy_time(fig, ax, data['COM_sheet'][map,0], data['COM_sheet'][map,1], speed[map], 'Speed $[Å/ps]$', cmap = 'plasma')
    plt.axis('equal')
    plt.xlabel(r'$\Delta COM_\parallel$ $[Å]$', fontsize=14)
    plt.ylabel(r'$\Delta COM_\perp$ $[Å]$', fontsize=14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    if save:
        plt.savefig('../article/figures/baseline/COM_path_K0.pdf', bbox_inches='tight')
        
    
    # fig = plt.figure(num = unique_fignum(), figsize = (10,3), dpi = 80, facecolor='w', edgecolor='k')
    # plt.plot(VA_pos[map], Ff_savgol[map], label = "$F_\parallel$", color = color_cycle(0))
    # plt.plot(VA_pos[map], Ff_perp_savgol[map], label = "$F_\perp$", color = color_cycle(1))
    
   


def mean_values(filename, save = False):
    # Parameters 
    mean_window_pct = 0.5 # relative length of the mean window [% of total duration]
    std_window_pct = 0.35  # relative length of the std windoe [% of mean window]
    
    
    info, data = analyse_friction_file(filename, mean_window_pct, std_window_pct)    
    time = data['time'] - data['time'][0]
    VA_pos = time * info['drag_speed']  # virtual atom position
    
    
    Ff = data[f'Ff_full_sheet'][:,0]
    map = [VA_pos <= 10][0]
    mean_window = int(mean_window_pct*len(time[map]))
    std_window = int(std_window_pct*mean_window)
    
    
    
    # --- 10 Å --- #
    # Running mean
    runmean, _ = running_mean(Ff[map], mean_window)

    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(VA_pos[map], Ff[map], label = "Raw data", color = color_cycle(0))
    plt.plot(VA_pos[map], runmean, label = f"Running mean ({int(mean_window_pct*100)}% window)", color = color_cycle(1))
    plt.plot(VA_pos[map][-1], runmean[-1], 'o', label = f'Final mean = {runmean[-1]:0.4f}', color = color_cycle(1))
  
    plt.xlabel(r'Sliding distance [Å]', fontsize=14)
    plt.ylabel(r'Friction force $F_\parallel$ [nN]', fontsize=14)
    plt.legend(loc = 'lower left', fontsize = 13)
    
    add_xaxis(plt.gca(), x = VA_pos[map], xnew = time[map], xlabel = 'Time [ps]', decimals = 0, fontsize = 14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
        plt.savefig('../article/figures/baseline/Ff_runmean.pdf', bbox_inches='tight')
    
    
    # Running std
    _, runmean_std = running_mean(runmean, std_window)
    runmean_std /= np.abs(runmean[-1])
    
    
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(VA_pos[map], runmean_std, color = color_cycle(2))
    plt.plot(VA_pos[map][-1], runmean_std[-1], 'o', label = f'Final estimate = {runmean_std[-1]:.3f}', color = color_cycle(2))
  
    plt.xlabel(r'Sliding distance [Å]', fontsize=14)
    plt.ylabel(r'Running rel. error', fontsize=14)
    plt.legend(fontsize = 13)
    
    add_xaxis(plt.gca(), x = VA_pos[map], xnew = time[map], xlabel = 'Time [ps]', decimals = 0, fontsize = 14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
        plt.savefig('../article/figures/baseline/Ff_runstd.pdf', bbox_inches='tight')
    
    
    # --- 400 Å --- #
    # Running std
    mean_window = int(mean_window_pct*len(time))
    std_window = int(std_window_pct*mean_window)
    
    runmean, _ = running_mean(Ff, mean_window)
    _, runmean_std = running_mean(runmean, std_window)
    runmean_std /= np.abs(runmean[-1])
    
    
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(VA_pos, runmean_std, color = color_cycle(2))
    plt.plot(VA_pos[-1], runmean_std[-1], 'o', label = f'Final estimate = {runmean_std[-1]:.3f}', color = color_cycle(2))
  
    plt.xlabel(r'Sliding distance [Å]', fontsize=14)
    plt.ylabel(r'Running rel. error', fontsize=14)
    plt.legend(fontsize = 13)
    
    add_xaxis(plt.gca(), x = VA_pos, xnew = time, xlabel = 'Time [ps]', decimals = 0, fontsize = 14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
        plt.savefig('../article/figures/baseline/Ff_runstd_long.pdf', bbox_inches='tight')
    
    
    print(data['Ff_std'])

def max_values(folder, save = False):
    """ Raw data """
    mean_window_pct = 0.5 # relative length of the mean window [% of total duration]
    std_window_pct = 0.35  # relative length of the std windoe [% of mean window]
    
    topn = 3
    
    # maxsize = 200
    # minsize = 50
    markers = ['o', 'v', 'D']
    sizes = [200, 100, 25]
    
    cmap = matplotlib.cm.viridis
    colorbar_scale = 'log'
    
    
    F_N = []
    argtop = []
    Ffmax = []
    
    for i, job_dir in enumerate(get_dirs_in_path(folder, sort = True)):
        try: # If info file exist
            info_dict = read_info_file(os.path.join(job_dir,'info_file.txt'))
            
            
            if 'is_ruptured' in info_dict:
                is_ruptured = info_dict['is_ruptured']
                
            else:
                print("Sim not done")
                continue
            
            
            F_N.append(metal_to_SI(info_dict['F_N'], 'F')*1e9)
            
            if not is_ruptured:
                # Get data
                friction_file = find_single_file(job_dir, ext = 'Ff.txt')     
                _, fricData = analyse_friction_file(friction_file, mean_window_pct, std_window_pct)
                
                
                time = fricData['time'] - fricData['time'][0]
                VA_pos = time * info_dict['drag_speed']  # virtual atom position
    
                
                
                
                Ff = fricData['Ff_full_sheet'][:,0]
                sort = np.flip(np.argsort(Ff))[:topn]
                # argtop = VA_pos[sort]
                
                argtop.append(VA_pos[sort])
                Ffmax.append(Ff[sort])
                
                # color = get_color_value(z[k], np.min(z), np.max(z), scale = colorbar_scale, cmap = cmap)
                # for n in range(len(sort)):
                #     size = minsize + (topn-n)*(maxsize-minsize)/topn
                #     plt.scatter(VA_pos[sort][n], Ff[sort][n], marker=f'${n+1}$', s = size, color = color_cycle(i) )

            else:
                continue
                    
        except FileNotFoundError:
            print(f"<-- Missing file")
    
    F_N = np.array(F_N)
    argtop = np.array(argtop)
    Ffmax = np.array(Ffmax)
    
    argsort = np.argsort(argtop, axis = 1)
    
   
    
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    for i in range(len(F_N)):
        color = get_color_value(F_N[i], np.min(F_N), np.max(F_N), scale = colorbar_scale, cmap = cmap)  
               
        # for n in reversed(range(len(argtop[i]))):
        for n in range(len(argtop[i])):
            # size = minsize + (topn-n)*(maxsize-minsize)/topn
            # plt.scatter(argtop[i ,n], Ffmax[i ,n], marker=f'${n+1}$', s = size, color = color)
            
            
            plt.scatter(argtop[i ,n], Ffmax[i ,n], marker=markers[n], s = sizes[n], alpha = 1, color = color, edgecolor = 'black', zorder = n)
            
        # Plot connections
        plt.plot(argtop[i][argsort[i]], Ffmax[i][argsort[i]], '--', linewidth = 0.5, alpha = 0.2, color = color, zorder = 0)
    
    
    if colorbar_scale == 'linear':
        norm = matplotlib.colors.BoundaryNorm(F_N, cmap.N)
    elif colorbar_scale == 'log':
        norm = matplotlib.colors.LogNorm(np.min(F_N), np.max(F_N))
    else:
        exit(f'scale = \'{colorbar_scale}\' is not defined.')
   
   
    # vline(plt.gca(), 71, linestyle = '--', color = 'black', linewidth = 1, zorder = 0, label = "Slow period $= 71 \pm 15$ Å")
    vline(plt.gca(), 71, linestyle = '--', color = 'black', linewidth = 1, zorder = 0, label = "$71 \pm 15$ Å")
    
     # Add scatter legend
    for n in range(argtop.shape[1]):
        plt.scatter([], [], color = 'grey', marker = markers[n], s = sizes[n], label = f'{n+1}')
    # h, l = ax.get_legend_handles_labels()
    # legend = ax.legend(h, l, loc='upper right',  handletextpad=0.00, fontsize = 13)
    # legend.set_title("Depth", {'size': 13})
    
    
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plt.fill_betweenx([Ffmax.min() - 20, Ffmax.max() + 20], [71-15, 71-15], [71+15, 71+15], color = 'black', alpha = 0.1, zorder = 0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
   
   
    cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
    cb.set_label(label = '$F_N$ [nN]', fontsize=14)
    plt.xlabel('Sliding distance [Å]', fontsize = 14)
    plt.ylabel(r'$\max \ F_\parallel$ [nN]', fontsize = 14)    
    plt.legend(fontsize = 13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
        plt.savefig('../article/figures/baseline/max_dist.pdf', bbox_inches='tight')
   
    
    
def maxarg_vs_K(dirs, save = False):
    mean_window_pct = 0.5 # relative length of the mean window [% of total duration]
    std_window_pct = 0.35  # relative length of the std windoe [% of mean window]
    
    K = np.zeros(len(dirs))
    argmax = np.zeros(len(dirs))
    
    for i, dir in enumerate(dirs):
        friction_file = find_single_file(dir, ext = 'Ff.txt')     
        info, data = analyse_friction_file(friction_file, mean_window_pct, std_window_pct)
        
        time = data['time'] - data['time'][0]
        VA_pos = time * info['drag_speed']  # virtual atom position

        
        K[i] = info['K']*metal_to_SI(1, 'F')/metal_to_SI(1, 's')
        
        if K[i] == 0:
            K[i] = 250
        
          
        Ffmax = data['Ff_full_sheet'][:,0] # max
        argmax[i] = VA_pos[np.argmax(Ffmax)]
        
    
    
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(K, argmax, 'o')
    hline(plt.gca(), 71, linestyle = '--', color = 'black', linewidth = 1, zorder = 0, label = "Slow period $= 71 \pm 15$ Å")
    
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plt.fill_between([K.min() - 20, K.max() + 20], [71-15, 71-15], [71+15, 71+15], color = 'black', alpha = 0.1, zorder = 0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    
    # xfill(plt.gca(), 71-15, 71+15, color = 'black', alpha = 0.2, linewidth = 1, zorder = 0)
    
    
    plt.xlabel(r'Spring constant $K$ [N/m]', fontsize=14)
    plt.ylabel(r'$\arg \min{F_{\parallel}}$ [Å]', fontsize=14)
    plt.legend(loc = 'upper right', fontsize = 13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
        plt.savefig('../article/figures/baseline/max_vs_K', bbox_inches='tight')
    


if __name__ == '__main__':
    path = '../Data/Baseline'
    # filename = os.path.join(path,'nocut/temp/T300/system_2023-01-17_Ff.txt')
    # filename = os.path.join(path,'nocut/vel/v1/system_v1_Ff.txt')
    filename = os.path.join(path,'nocut/special/v1/system_v1_Ff.txt')
    
    # path = '../Data/Baseline_fixmove' # XXX
    # filename = os.path.join(path,'nocut/temp/T300/system_T300_Ff.txt') # XXX
    # filename = os.path.join(path,'nocut/spring/K10/system_K10_Ff.txt')
    
    
    raw_data(filename, save = False)
    # ft(filename, save = False)
    # decomp(filename, save = False)
    # COM(filename, save = False)
    # mean_values(filename, save = False)
    # plt.show()
    
    #############################################
    
    # folder = os.path.join(path,'nocut/multi_stretch/stretch_15001_folder')
    # folder = os.path.join(path,'nocut/multi_FN/stretch_15001_folder')
    # max_values(folder, save = True)
    
    
    # path = '../Data/Baseline_fixmove'
    # files = get_dirs_in_path(os.path.join(path, 'nocut/spring'))
    # maxarg_vs_K(files, save = False)
    
    
    
    plt.show()


    