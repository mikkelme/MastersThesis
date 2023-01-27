import sys
sys.path.append('../') # parent folder: MastersThesis
import matplotlib.pyplot as plt
import numpy as np
from plot_set import *
from analysis.analysis_utils import *



def raw_data(filename, save = False):
    """ Raw data """
    
    mean_window_pct = 0.5 # relative length of the mean window [% of total duration]
    std_window_pct = 0.2  # relative length of the std windoe [% of mean window]
    
    
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
    
    



def ft(filename, save = False):
    """ Fourier transform """
    mean_window_pct = 0.5 # relative length of the mean window [% of total duration]
    std_window_pct = 0.2  # relative length of the std windoe [% of mean window]
    
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
    
    plt.xlabel(r'Drag length [Å]', fontsize=14)
    # plt.xlabel(r'Time [ps]', fontsize=14)
    plt.ylabel(r'Friction force $F_\parallel$ [nN]', fontsize=14)
    plt.legend(loc = 'lower left', fontsize = 13)
    
    add_xaxis(plt.gca(), x = VA_pos[map], xnew = time[map], xlabel = 'Time [ps]', decimals = 0, fontsize = 14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
        plt.savefig('../article/figures/baseline/ft_sine.pdf', bbox_inches='tight')
    
    
def decomp(filename, save = False):
    mean_window_pct = 0.5 # relative length of the mean window [% of total duration]
    std_window_pct = 0.2  # relative length of the std windoe [% of mean window]
    
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
    
    plt.xlabel(r'Drag length [Å]', fontsize=14)
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
        
    plt.xlabel(r'Drag length [Å]', fontsize=14)
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
    # plt.xlabel(r'Drag length [Å]', fontsize=14)
    # plt.ylabel(r'---', fontsize=14)
    # plt.legend(fontsize = 13)
    # plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    

    # # plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    # # plt.plot(VA_pos[map], Ff_perp_savgol[map], label = "Ff perp")
    # # plt.plot(VA_pos[map], move_perp_savgol[map], label = "move perp force")
    # # plt.xlabel(r'Drag length [Å]', fontsize=14)
    # # plt.ylabel(r'---', fontsize=14)
    # # plt.legend(fontsize = 13)
    # # plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    

def COM(filename, save = False):
    mean_window_pct = 0.5 # relative length of the mean window [% of total duration]
    std_window_pct = 0.2  # relative length of the std windoe [% of mean window]

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
    # path = '../Data/Baseline'
    # baseline1(os.path.join(path,'nocut/temp/T300/system_2023-01-17_Ff.txt'), save = False)
    # baseline2(os.path.join(path,'nocut/temp/T300/system_2023-01-17_Ff.txt'), save = False) # Get some more ill bhaving data here. 
   
    path = '../Data/Baseline_fixmove'
    filename = os.path.join(path,'nocut/temp/T300/system_T300_Ff.txt')
    # raw_data(filename, save = False)
    # ft(filename, save = False)
    # decomp(filename, save = False)
    COM(filename, save = True)
    # baseline2(os.path.join(path,'nocut/temp/T300/system_T300_Ff.txt'), save = False) # Get some more ill bhaving data here. 
    
    plt.show()


    