import sys
sys.path.append('../') # parent folder: MastersThesis
import matplotlib.pyplot as plt
import numpy as np
from plot_set import *
from analysis.analysis_utils import *


def baseline1(filename, save = False):
    """ Analyse a single friction measurement """
    
    mean_window_pct = 0.5 # relative length of the mean window [% of total duration]
    std_window_pct = 0.2  # relative length of the std windoe [% of mean window]
    
    
    info, data = analyse_friction_file(filename, mean_window_pct, std_window_pct)    
    time = data['time'] - data['time'][0]
    VA_pos = time * info['drag_speed']  # virtual atom position
    
 
    # --- Figure 1 --- #
    # (VA_pos, Ff full sheet parallel) | drag length = 10 Å
    Ff = data[f'Ff_full_sheet'][:,0]
    
    map = [VA_pos <= 10][0]
    window_length = 150; polyorder = 5
    print(f'window length = {window_length}, corresponding to drag distance {VA_pos[window_length]} Å and time {time[window_length]} ps')
    Ff_savgol = savgol_filter(window_length, polyorder, Ff)[0]
    
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(VA_pos[map], Ff[map], label = "Raw data")
    plt.plot(VA_pos[map], Ff_savgol[map], label = f"Savgol filter")
    plt.xlabel(r'Drag length [Å]', fontsize=14)
    plt.ylabel(r'Friction force $F_\parallel$ [nN]', fontsize=14)
    plt.legend(loc = 'lower left', fontsize = 13)
    
    add_xaxis(plt.gca(), x = VA_pos[map], xnew = time[map], xlabel = 'Time [ps]', decimals = 0, fontsize = 14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
        plt.savefig('../article/figures/baseline/drag_Ff_10Å.pdf', bbox_inches='tight')
    
    
    # --- Figure 2 --- #
    # (VA_pos, Ff full sheet parallel) | drag length = 50 Å
    map = [VA_pos <= 100][0]
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(VA_pos[map], Ff[map], label = "Raw data")
    plt.plot(VA_pos[map], Ff_savgol[map], label = f"Savgol filter")
    plt.xlabel(r'Drag length [Å]', fontsize=14)
    plt.ylabel(r'Friction force $F_\parallel$ [nN]', fontsize=14)
    plt.legend(loc = 'lower left', fontsize = 13)
    
    add_xaxis(plt.gca(), x = VA_pos[map], xnew = time[map], xlabel = 'Time [ps]', decimals = 0, fontsize = 14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
        plt.savefig('../article/figures/baseline/drag_Ff_100Å.pdf', bbox_inches='tight')
    
    # --- Figure 3-6 --- #
    # Fourier transform 
    Ff_fourier = np.fft.fft(Ff)/len(Ff)  # Normalize
    Ff_fourier = Ff_fourier[range(int(len(Ff)/2))] # Exclude sampling frequency
    
    freq = np.arange(int(len(VA_pos)/2))/VA_pos[-1]
    amplitude = abs(Ff_fourier)
    
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(freq, amplitude)
    plt.xlabel(r'Frequency [1/Å]', fontsize=14)
    plt.ylabel(r'Amplitude', fontsize=14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
        plt.savefig('../article/figures/baseline/ft.pdf', bbox_inches='tight')
    
    # Zoom and pick out interestingfrequencies
    zoom = len(freq)//15
    peaks = [11, 158, 305, 453] 
    
    
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(freq[:zoom], amplitude[:zoom])
    plt.xlabel(r'Frequency [1/Å]', fontsize=14)
    plt.ylabel(r'Amplitude', fontsize=14)
    
    for idx in peaks:
        plt.plot(freq[idx], amplitude[idx], 'ko')
        plt.text(freq[idx] + 0.005, amplitude[idx], f'$({freq[idx]:.3f})$', fontsize = 14)
    if save:
        plt.savefig('../article/figures/baseline/ft_zoom.pdf', bbox_inches='tight')
    
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
 
    # Apply frequencis to orginal plot
    map = [VA_pos <= 10][0]
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(VA_pos[map], Ff[map], label = "Raw data")
    plt.plot(VA_pos[map], 0.7*np.sin(2*np.pi*freq[peaks[1]]*VA_pos[map]), label = f'freq. = {freq[peaks[1]]:.3f}/Å', color = color_cycle(3))
    
    plt.xlabel(r'Drag length [Å]', fontsize=14)
    plt.ylabel(r'Friction force $F_\parallel$ [nN]', fontsize=14)
    plt.legend(loc = 'lower left', fontsize = 13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
        plt.savefig('../article/figures/baseline/ft_sine_zoom.pdf', bbox_inches='tight')
    
    map = [VA_pos <= 100][0]
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(VA_pos[map], Ff[map], label = "Raw data")
    # plt.plot(VA_pos[map], 1.25 + 0.4*np.sin(2*np.pi*freq[peaks[0]]*VA_pos[map] - 0.5*np.pi), label = f'freq. = {freq[peaks[0]]:.3f}', color = color_cycle(4))
    plt.plot(VA_pos[map],  0.7*np.sin(2*np.pi*freq[peaks[0]]*VA_pos[map] - 0.5*np.pi), label = f'freq. = {freq[peaks[0]]:.3f}/Å', color = color_cycle(4))
    
    plt.xlabel(r'Drag length [Å]', fontsize=14)
    plt.ylabel(r'Friction force $F_\parallel$ [nN]', fontsize=14)
    plt.legend(loc = 'lower left', fontsize = 13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
        plt.savefig('../article/figures/baseline/ft_sine.pdf', bbox_inches='tight')
    

    
    # # --- Figure 7 --- #
    # # Friction and move force 
    # move = data['move_force'][:, 0]
    # move_perp = data['move_force'][:, 0]
    # move_savgol, move_perp_savgol = savgol_filter(window_length, polyorder, move, move_perp)
    # Ff_perp = data[f'Ff_full_sheet'][:,1]
    # Ff_perp_savgol = savgol_filter(window_length, polyorder, Ff_perp)[0]

    # plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    # plt.plot(VA_pos[map], Ff_savgol[map], label = "Ff")
    # plt.plot(VA_pos[map], move_savgol[map], label = "move force")
    # plt.xlabel(r'Drag length [Å]', fontsize=14)
    # plt.ylabel(r'---', fontsize=14)
    # plt.legend(fontsize = 13)
    # plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    

    # plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    # plt.plot(VA_pos[map], Ff_perp_savgol[map], label = "Ff perp")
    # plt.plot(VA_pos[map], move_perp_savgol[map], label = "move perp force")
    # plt.xlabel(r'Drag length [Å]', fontsize=14)
    # plt.ylabel(r'---', fontsize=14)
    # plt.legend(fontsize = 13)
    # plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    
    # --- Figure 7-8 --- #
    # Force decomposition: Parallel and perpendicular
    Ff_perp = data[f'Ff_full_sheet'][:,1]
    Ff_perp_savgol = savgol_filter(window_length, polyorder, Ff_perp)[0]
    
    # Para and perp
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(VA_pos[map], Ff_savgol[map], label = "$F_\parallel$", color = color_cycle(0))
    plt.plot(VA_pos[map], Ff_perp_savgol[map], label = "$F_\perp$", color = color_cycle(1))
    
    plt.xlabel(r'Drag length [Å]', fontsize=14)
    plt.ylabel(r'Friction force $F_\parallel$ [nN]', fontsize=14)
    plt.legend(loc = 'lower left', fontsize = 13)
    
    add_xaxis(plt.gca(), x = VA_pos[map], xnew = time[map], xlabel = 'Time [ps]', decimals = 0, fontsize = 14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    

    if save:
        plt.savefig('../article/figures/baseline/decomp_direc.pdf', bbox_inches='tight')
        
    # # Norm
    # Ff_norm_savgol =  np.sqrt(Ff_savgol**2 + Ff_perp_savgol**2) # calculating the norm after savgol makes the curve a lot closer to expected trend
    
    # plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    # plt.plot(VA_pos[map], Ff_norm_savgol[map], label = "$Norm ||F||$", color = color_cycle(2))
    
    # plt.xlabel(r'Drag length [Å]', fontsize=14)
    # plt.ylabel(r'Friction force $F_\parallel$ [nN]', fontsize=14)
    # plt.legend(loc = 'lower left', fontsize = 13)
    
    # add_xaxis(plt.gca(), x = VA_pos[map], xnew = time[map], xlabel = 'Time [ps]', decimals = 0, fontsize = 14)
    # plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    # if save:
    #     plt.savefig('../article/figures/baseline/decomp_direc_norm.pdf', bbox_inches='tight')
        


    # --- Figure 9 --- #
    # Force decomposition: Full sheet = sheet + PB
    
    Ff_sheet =  data[f'Ff_sheet'][:,1]
    Ff_PB =  data[f'Ff_PB'][:,1]
    Ff_full = Ff_sheet + Ff_PB
    Ff_sheet_savgol, Ff_PB_savgol, Ff_full_savgol = savgol_filter(window_length, polyorder, Ff_sheet, Ff_PB, Ff_full)
    
    # Ff_full_savgol = Ff_sheet_savgol + Ff_PB_savgol
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(VA_pos[map], Ff_sheet_savgol[map], label = "Sheet")
    plt.plot(VA_pos[map], Ff_PB_savgol[map], label = "PB")
    # plt.plot(VA_pos[map], Ff_full_savgol[map], label = "Ff full")
    
    plt.xlabel(r'Drag length [Å]', fontsize=14)
    plt.ylabel(r'Friction force $F_\parallel$ [nN]', fontsize=14)
    plt.legend(loc = 'lower left', fontsize = 13)
    
    add_xaxis(plt.gca(), x = VA_pos[map], xnew = time[map], xlabel = 'Time [ps]', decimals = 0, fontsize = 14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
         plt.savefig('../article/figures/baseline/decomp_group.pdf', bbox_inches='tight')
   


    # --- Figure 10 --- #
    # COM oath
    
    map = [VA_pos <= 10][0]
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
        plt.savefig('../article/figures/baseline/COM_path.pdf', bbox_inches='tight')
   


def baseline2(filename, save = False):
    """ Analyse a single friction measurement """
    
    # Parameters 
    mean_window_pct = 0.5 # relative length of the mean window [% of total duration]
    std_window_pct = 0.2  # relative length of the std windoe [% of mean window]
    
    
    info, data = analyse_friction_file(filename, mean_window_pct, std_window_pct)    
    time = data['time'] - data['time'][0]
    VA_pos = time * info['drag_speed']  # virtual atom position
    
    
    # --- Uncertainty --- #
    Ff = data[f'Ff_full_sheet'][:,0]
    map = [VA_pos <= 10][0]
    mean_window = int(mean_window_pct*len(time[map]))
    std_window = int(std_window_pct*mean_window)
    
    
    
    # Running mean 
    runmean, _ = running_mean(Ff[map], mean_window)

    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(VA_pos[map], Ff[map], label = "Raw data")
    plt.plot(VA_pos[map], runmean, label = f"Running mean ({int(mean_window_pct*100)}% window)")
  
    plt.xlabel(r'Drag length [Å]', fontsize=14)
    plt.ylabel(r'Friction force $F_\parallel$ [nN]', fontsize=14)
    plt.legend(loc = 'lower left', fontsize = 13)
    
    add_xaxis(plt.gca(), x = VA_pos[map], xnew = time[map], xlabel = 'Time [ps]', decimals = 0, fontsize = 14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
        plt.savefig('../article/figures/baseline/Ff_running_mean.pdf', bbox_inches='tight')
    
    
    # Runing std
    runmean_mean, runmean_std = running_mean(runmean, std_window)
    runmean_std /= np.abs(runmean_mean)
    
    plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(VA_pos[map], runmean_std, label = "Running std")
  
    plt.xlabel(r'Drag length [Å]', fontsize=14)
    plt.ylabel(r'Running relative std [nN]$ [nN]', fontsize=14)
    plt.legend(fontsize = 13)
    
    add_xaxis(plt.gca(), x = VA_pos[map], xnew = time[map], xlabel = 'Time [ps]', decimals = 0, fontsize = 14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
        plt.savefig('../article/figures/baseline/Ff_running_std.pdf', bbox_inches='tight')
    
 
   
    print(data['Ff_std'])
   


if __name__ == '__main__':
    path = '../Data/Baseline'
    # baseline1(os.path.join(path,'nocut/temp/T300/system_2023-01-17_Ff.txt'), save = True)
    baseline2(os.path.join(path,'nocut/temp/T300/system_2023-01-17_Ff.txt'), save = False) # Get some more ill bhaving data here. 
    
    plt.show()


    