### Some scripts used in the initial data analysis
### these are not used for the actual plotting
### in the thesis. For that see /produce_figures
from analysis_utils import *


def drag_length_dependency(filename):
    data = analyse_friction_file(filename)    
    time = data['time']
    COM = data['COM_sheet'][:,0]
    contact = data['contact'][:,0]
    
    info = read_info_file('/'.join(filename.split('/')[:-1]) + '/info_file.txt' )
    VA_pos = (time - time[0]) * info['drag_speed']  # virtual atom position
    
   
    # mean_window = int(np.argmin(np.abs(VA_pos - 50)))
    runmean_wpct = 0.5
    runstd_wpct = 0.2
    mean_window = int(runmean_wpct*len(time))
    std_window = int(runstd_wpct*mean_window)
    
    
    group_name = {0: 'full_sheet', 1: 'sheet', 2: 'PB'}
    for g in reversed(range(1)):
        
        fig2 = plt.figure(num = unique_fignum())
        fig2.suptitle(filename)
        grid = (1,2)
        ax3 = plt.subplot2grid(grid, (0, 0), colspan=1)
        ax4 = plt.subplot2grid(grid, (0, 1), colspan=1)
        
       
        fig1 = plt.figure(num = unique_fignum())
        fig1.suptitle(filename)
        
        grid = (1,2)
        ax1 = plt.subplot2grid(grid, (0, 0), colspan=1)
        ax2 = plt.subplot2grid(grid, (0, 1), colspan=1)
        
        
        Ff = data[f'Ff_{group_name[g]}'][:,0]
        
        # Ff = data['move_force'][:,0]
        # A = data[f'Ff_{group_name[g]}'][:,0]
        # B = data[f'Ff_{group_name[g]}'][:,1]
        # Ff = np.sqrt(A**2 + B**2)
        
        runmean = running_mean(Ff, window_len = mean_window)[0]
        rel_std = running_mean(runmean, window_len = std_window)[1]/runmean[~np.isnan(runmean)][-1]
        
        ax1.plot(VA_pos, Ff, label = f'Ydata ({group_name[g]})')
        ax1.plot(VA_pos, cum_mean(Ff), label = "Cum mean")
        ax1.plot(VA_pos, runmean, label = "running mean")
        ax1.plot(VA_pos, cum_max(Ff), label = "Cum max")
        # ax1.plot(VA_pos, cumTopQuantileMax(Ff, quantile, brute_force = False), label = f"Top {quantile*100}% max mean")
        ax1.set(xlabel='VA pos $\parallel$ [Å]', ylabel='$F_\parallel$ [nN]')
        ax1.legend(loc = 'lower center', fontsize = 10, ncol=2, fancybox = True, shadow = True)
              
        ax3.plot(VA_pos, rel_std, label = "Ff std")
        ax3.set(xlabel='VA pos $\parallel$ [Å]', ylabel='Ff rel. runmean std')
        runmean = running_mean(contact, window_len = mean_window)[0]
        rel_std = running_mean(runmean, window_len = std_window)[1]/runmean[~np.isnan(runmean)][-1]
        
        
        ax2.plot(VA_pos, contact, label = "Ydata (full sheet)")
        ax2.plot(VA_pos, cum_mean(contact), label = "Cum mean")
        ax2.plot(VA_pos, runmean, label = "running mean")
        ax2.legend(loc = 'lower center', fontsize = 10, ncol=2, fancybox = True, shadow = True)
        ax2.set(xlabel='VA pos $\parallel$ [Å]', ylabel='Contact (Full sheet) [%]')
        
        ref = runmean[~np.isnan(runmean)][-1]
        ax4.plot(VA_pos, rel_std, label = "Contact std")
        ax4.set(xlabel='VA pos $\parallel$ [Å]', ylabel='Contact rel. runmean std')

        
    # for ax in [ax1, ax2]:
    #     add_xaxis(ax, time, VA_pos, xlabel='COM$\parallel$ [Å]', decimals = 1)    
    
    for fig in [fig1, fig2]:
        fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    return [interactive_plotter(fig) for fig in [fig1, fig2]]
    
    
def drag_length_compare(filenames):
    group_name = {0: 'full_sheet', 1: 'sheet', 2: 'PB'}
    
    xaxis = "VA_pos" # 'time' || 'COM'
    relative = False
    
    g = 0
        
    quantile = 0.999
    runmean_wpct = 0.5
    runstd_wpct = 0.2
   
    fig1 = plt.figure(figsize = (6, 6), num = 0)
    fig1.suptitle("Mean $F_\parallel$ [nN]")
    
    grid = (2,2)
    ax1 = plt.subplot2grid(grid, (0, 0), colspan=1)
    ax2 = plt.subplot2grid(grid, (0, 1), colspan=1)
    ax3 = plt.subplot2grid(grid, (1, 0), colspan=1)
    ax4 = plt.subplot2grid(grid, (1, 1), colspan=1)
    
    
    fig2 = plt.figure(figsize = (6, 6), num = 1)
    fig2.suptitle("Max $F_\parallel$ [nN]")
    grid = (2,2)
    ax5 = plt.subplot2grid(grid, (0, 0), colspan=1)
    ax6 = plt.subplot2grid(grid, (0, 1), colspan=1)
    ax7 = plt.subplot2grid(grid, (1, 0), colspan=1)
    ax8 = plt.subplot2grid(grid, (1, 1), colspan=1)
    
    fig3 = plt.figure(figsize = (6, 6), num = 2)
    fig3.suptitle("Mean contact [%]")
    grid = (2,2)
    ax9 = plt.subplot2grid(grid, (0, 0), colspan=1)
    ax10 = plt.subplot2grid(grid, (0, 1), colspan=1)
    ax11 = plt.subplot2grid(grid, (1, 0), colspan=1)
    ax12 = plt.subplot2grid(grid, (1, 1), colspan=1)
    
    for i, filename in enumerate(filenames):
        print(f"\rFile: ({i+1}/{len(filenames)}) || Reading data | {filename}       ", end = " ")
        # if i == 3:
        #     print()
        #     exit("HERE")
        data = analyse_friction_file(filename)   
        COM = data['COM_sheet'][:,0]
        time = data['time']
        info = read_info_file('/'.join(filename.split('/')[:-1]) + '/info_file.txt' )
        VA_pos = (time - time[0]) * info['drag_speed']  # virtual atom position
        contact = data['contact'][:,0]
        Ff = data[f'Ff_{group_name[g]}'][:,0]
        
        mean_window = int(runmean_wpct*len(time))
        std_window = int(runstd_wpct*mean_window)
    
        
        if xaxis == 'time':
            x = time; xlabel = 'Time [ps]'
        elif xaxis == 'VA_pos':
            x = VA_pos; xlabel = 'VA pos $\parallel$ [Å]'
        else:
            print(f'xaxis = {xaxis} is not a known setting.')
        
        if relative:
            xlabel = 'Rel ' + xlabel
            x /= x[-1] # relative drag
        
        # --- Ff (mean) --- # 
        print(f"\rFile: ({i+1}/{len(filenames)}) | Mean Ff | {filename}       ", end = " ")
        cummean = cum_mean(Ff)
        cummean_rel_runstd = running_mean(cummean, window_len = std_window)[1]/cummean[~np.isnan(cummean)][-1]
        
        runmean = running_mean(Ff, window_len = mean_window)[0]
        runmean_rel_runstd = running_mean(runmean, window_len = std_window)[1]/runmean[~np.isnan(runmean)][-1]
        
        # Mean
        ax1.plot(x, cummean)
        ax2.plot(x, cummean_rel_runstd, label = filename)
        ax1.set(xlabel=xlabel, ylabel='cum mean')
        ax2.set(xlabel=xlabel, ylabel='Rel. runmean std')
        
        # Running mean 
        ax3.plot(x, runmean)
        ax4.plot(x, runmean_rel_runstd)
        ax3.set_xlim(ax1.get_xlim())
        ax4.set_xlim(ax2.get_xlim())
        ax3.set(xlabel=xlabel, ylabel=f'run mean ({runmean_wpct*100:.0f}%)')
        ax4.set(xlabel=xlabel, ylabel='Rel. runmean std')
        
            
        # --- Ff (max) --- #  (not updated)
        print(f"\rFile: ({i+1}/{len(filenames)}) | Max Ff | {filename}       ", end = " ")
        cummean_topmax = cumTopQuantileMax(np.abs(data[f'Ff_{group_name[g]}'][:,0]), quantile)
        cummean_rel_runstd = running_mean(cummean_topmax, window_len = std_window)[1]/cummean_topmax[~np.isnan(cummean_topmax)][-1]
        # _, cummean_topmax_runstd = running_mean(cummean_topmax, window_len = int(runstd_wpct*len(cummean_topmax)))
        
        cummax = cum_max(data[f'Ff_{group_name[g]}'][:,0])
        cummax_rel_runstd = running_mean(cummax, window_len = std_window)[1]/cummax[~np.isnan(cummax)][-1]
        # _, cummax_runstd = running_mean(cummax, window_len = int(runstd_wpct*len(cummax)))


       
        # Mean top quantile max
        ax5.plot(x, cummean_topmax)
        ax6.plot(x, cummean_rel_runstd, label = filename)
        ax5.set(xlabel=xlabel, ylabel='cum mean top max')
        ax6.set(xlabel=xlabel, ylabel=f'cummean topmax rel. std')
       
        # Max 
        ax7.plot(x, cummax)
        ax8.plot(x, cummax_rel_runstd)
        ax7.set(xlabel=xlabel, ylabel='cum max')
        ax8.set(xlabel=xlabel, ylabel=f'cummax topmax rel std')
        
    
    
        # --- Contact --- #
        print(f"\rFile: ({i+1}/{len(filenames)}) | Mean contact | {filename}       ", end = " ")
        cummean = cum_mean(contact)
        cummean_rel_runstd = running_mean(cummean, window_len = std_window)[1]/cummean[~np.isnan(cummean)][-1]
        
        runmean = running_mean(contact, window_len = mean_window)[0]
        runmean_rel_runstd = running_mean(runmean, window_len = std_window)[1]/runmean[~np.isnan(runmean)][-1]
        
            
        # Mean 
        ax9.plot(x, cummean, label = filename)
        ax10.plot(x, cummean_rel_runstd)
        ax9.set(xlabel=xlabel, ylabel='cum mean')
        ax10.set(xlabel=xlabel, ylabel='Rel. runmean std')
        
        # Running mean 
        ax11.plot(x, runmean)
        ax12.plot(x, runmean_rel_runstd)
        ax11.set_xlim(ax9.get_xlim())
        ax12.set_xlim(ax10.get_xlim())
        ax11.set(xlabel=xlabel, ylabel=f'run mean ({runmean_wpct*100:.0f}%)')
        ax12.set(xlabel=xlabel, ylabel='Rel. runmean std')
    
    print()
    # for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]:
    #     add_xaxis(ax, time, COM, xlabel='COM$\parallel$ [Å]', decimals = 1)    
    
    
    
    obj = []
    for fig in [fig1, fig2, fig3]:  
    
        # fig.subplots_adjust(bottom=0.3)
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.0), fontsize = 10, fancybox=True, shadow=False, ncol=1)
        fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2, rect = (0, 0.2, 1, 1))
        # fig.legend(loc = 'lower center', fontsize = 10, ncol=1, fancybox = True, shadow = True)
        obj.append(interactive_plotter(fig))
    return obj


def variable_dependency(filenames, variable_name = None, drag_cap = None):
    assert variable_name is not None, "Please define variable_name varying." 
    list_type = not isinstance(variable_name, str) and hasattr(variable_name, '__len__')
    if list_type:
        assert len(variable_name) == len(filenames), f"variable values provided through variable_name ({type(variable_name)}) with len {len(variable_name)} does not match the filenames with len {len(filenames)}"

    
    fig2 = plt.figure(figsize = (6, 6), num = unique_fignum())
    grid = (2,1)
    ax4 = plt.subplot2grid(grid, (0, 0), colspan=1)
    ax5 = plt.subplot2grid(grid, (1, 0), colspan=1)
    
    
    fig1 = plt.figure(figsize = (6, 6), num = unique_fignum())
    grid = (3,1)
    ax1 = plt.subplot2grid(grid, (0, 0), colspan=1)
    ax2 = plt.subplot2grid(grid, (1, 0), colspan=1)
    ax3 = plt.subplot2grid(grid, (2, 0), colspan=1)
    
    mean_pct = 0.5
    std_pct = 0.2
    
    linewidth = 1.5
    marker = 'o'
    markersize = 2.5
    rup_marker = 'x'
    rupmarkersize = markersize * 3
            

    Ffmax = np.zeros(len(filenames))
    Ffmean = np.zeros(len(filenames))
    Ff_std = np.zeros(len(filenames))
    contact = np.zeros(len(filenames))
    contact_std = np.zeros(len(filenames))
    variable = np.zeros(len(filenames))
    is_ruptured = np.zeros(len(filenames))
    
    for i, filename in enumerate(filenames):
        data = analyse_friction_file(filename, mean_pct, std_pct, drag_cap = drag_cap)  
        info = read_info_file('/'.join(filename.split('/')[:-1]) + '/info_file.txt' )
        time = data['time']
        Ff = data['Ff']
        VA_pos = (time - time[0]) * info['drag_speed']  # virtual atom position
        
        
        Ffmax[i] = Ff[0,0] # full sheet max
        Ffmean[i] = Ff[0, 1] # full sheet mean
        Ff_std[i] = data['Ff_std'][0] # ful sheet std from run mean
        contact[i] = data['contact_mean'][0]
        contact_std[i] = data['contact_std'][0] # full sheet std from run mean
        if list_type:
            variable[i] = variable_name[i]
        else:
            try:
                variable[i] = info[variable_name]
            except KeyError:
                print(f'key: {variable_name} not found in info. Valid keys are:')
                exit(info.keys())
                
        try:
            is_ruptured[i] = info['is_ruptured']
        except KeyError:
            is_ruptured[i] = 0
              
    
    sort = np.argsort(variable)
    variable = variable[sort]
    Ffmax = Ffmax[sort]
    Ffmean = Ffmean[sort]
    Ff_std = Ff_std[sort]
    contact = contact[sort]
    contact_std = contact_std[sort]
    is_ruptured = is_ruptured[sort]
    
    rup_true = np.argwhere(is_ruptured > 0)
    rup_false = np.argwhere(is_ruptured == 0)
    
    xlabel = variable_name
    if list_type:
        Ffmax /= variable**2
        Ffmean /= variable**2
        xlabel = 'sheet length' # TODO
    
    ax1.plot(variable, Ffmax, linewidth = linewidth, color = color_cycle(0))
    ax1.plot(variable[rup_true], Ffmax[rup_true], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color = color_cycle(0))
    ax1.plot(variable[rup_false], Ffmax[rup_false], linestyle = 'None', marker = marker, markersize = markersize, color = color_cycle(0))
    ax1.set(xlabel=xlabel, ylabel='max $F_\parallel$ $[eV/Å]$')
                
    ax2.plot(variable, Ffmean, linewidth = linewidth, color = color_cycle(1))
    ax2.plot(variable[rup_true], Ffmean[rup_true], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color = color_cycle(1))
    ax2.plot(variable[rup_false], Ffmean[rup_false], linestyle = 'None', marker = marker, markersize = markersize, color = color_cycle(1))
    ax2.set(xlabel=xlabel, ylabel='mean $F_\parallel$ $[eV/Å]$')
    
    ax3.plot(variable, contact, linewidth = linewidth, color = color_cycle(2))
    ax3.plot(variable[rup_true], contact[rup_true], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color = color_cycle(2))
    ax3.plot(variable[rup_false], contact[rup_false], linestyle = 'None', marker = marker, markersize = markersize, color = color_cycle(2))
    ax3.set(xlabel=xlabel, ylabel='Mean contact [%]"')
    
    fig1.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)


    # --- STD --- #
    
    
    ax4.plot(variable, Ff_std, linewidth = linewidth, color = color_cycle(1))
    ax4.plot(variable[rup_true], Ff_std[rup_true], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color = color_cycle(1))
    ax4.plot(variable[rup_false], Ff_std[rup_false], linestyle = 'None', marker = marker, markersize = markersize, color = color_cycle(1))
    ax4.set(xlabel=xlabel, ylabel='Ff std')
    
    ax5.plot(variable, contact_std, linewidth = linewidth, color = color_cycle(2))
    ax5.plot(variable[rup_true], contact_std[rup_true], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color = color_cycle(2))
    ax5.plot(variable[rup_false], contact_std[rup_false], linestyle = 'None', marker = marker, markersize = markersize, color = color_cycle(2))
    ax5.set(xlabel=xlabel, ylabel='Contact std')
    


if __name__ == "__main__":
    # Parrent folder
    path = '../Data/Baseline'
    
    # size = get_files_in_folder('../Data/Baseline/size', ext = 'Ff.txt') 
    # size_val = [round(np.sqrt(eval(s.split('_')[-2].replace('x','*')))) for s in size]
    
    # spring = get_files_in_folder('../Data/Baseline/spring', ext = 'Ff.txt')
    # temp = get_files_in_folder('../Data/Baseline/temp', ext = 'Ff.txt')
    # vel = get_files_in_folder('../Data/Baseline/vel', ext = 'Ff.txt')
    # dt = get_files_in_folder('../Data/Baseline/dt', ext = 'Ff.txt')
    
    # variable_dependency(size, variable_name = size_val)
    # variable_dependency(spring, variable_name = 'K')
    # variable_dependency(temp, variable_name = 'temp')
    # variable_dependency(vel, variable_name = 'drag_speed')
    
    
    
    # temp = get_files_in_folder('../Data/Baseline/honeycomb/temp', ext = 'Ff.txt')
    
    # drag_length_compare(temp)
    
    # temp = get_files_in_folder('../Data/Baseline/temp', ext = 'Ff.txt')
    # drag_length_compare(vel)
    # obj = drag_length_dependency(os.path.join(path,'nocut/temp/T5/system_2023-01-17_Ff.txt'))
  
  
  
    # single_measurement(os.path.join(path,'nocut/temp/T300/system_2023-01-17_Ff.txt'))
  
  
  
  
  
  
    #################################################################################
    # PF = "drag_length" 
    # PF = "drag_length_200nN" 
    # # PF = "drag_length_s200nN" 
    
    # ref = f'../Data/Baseline/{PF}/ref/system_ref_Ff.txt'
    
    # v05 = f'../Data/Baseline/{PF}/v05/system_v05_Ff.txt'
    # v5 = f'../Data/Baseline/{PF}/v5/system_v5_Ff.txt'
    # v10 = f'../Data/Baseline/{PF}/v10/system_v10_Ff.txt'
    # v20 = f'../Data/Baseline/{PF}/v20/system_v20_Ff.txt'
    # v50 = f'../Data/Baseline/{PF}/v50/system_v50_Ff.txt'
    # v100 = f'../Data/Baseline/{PF}/v100/system_v100_Ff.txt'
    
    # K0 = f'../Data/Baseline/{PF}/K0/system_K0_Ff.txt'
    # K5 = f'../Data/Baseline/{PF}/K5/system_K5_Ff.txt'
    # K10 = f'../Data/Baseline/{PF}/K10/system_K10_Ff.txt'
    
    # T5 = f'../Data/Baseline/{PF}/T5/system_T5_Ff.txt'
    # T50 = f'../Data/Baseline/{PF}/T50/system_T50_Ff.txt'
    # T200 = f'../Data/Baseline/{PF}/T200/system_T200_Ff.txt'
    # T300 = f'../Data/Baseline/{PF}/T300/system_T300_Ff.txt'
   
    # amorph = f'../Data/Baseline/{PF}/amorph/system_amorph_Ff.txt'
    # gold = f'../Data/Baseline/{PF}/gold/system_gold_Ff.txt'
   
    # vel_compare = [v05, ref, v5, v10, v20, v50, v100]
    # temp_compare = [T5, ref, T300]
    # K_compare = [K5, K10, ref, K0]
    # substrate_compare = [ref, amorph, gold]
    
    # v10_comp = [f'../Data/Baseline/{PF}/v10/system_v10_Ff.txt' for PF in ['drag_length', 
    #                                                                          'drag_length_200nN', 
    #                                                                          'drag_length_s200nN']]
    
    
    
    # custom_comp = [ '../Data/Multi/updated_LJ/ref1/stretch_15000_folder/job0/system_drag_Ff.txt',
    #                 '../Data/Multi/contact_area_cut130/cut130/stretch_15000_folder/job0/system_drag_Ff.txt']
    #             #    '../Data/Multi/nocuts/ref1/stretch_315000_folder/job2/system_drag_Ff.txt']
    
    
    # obj = drag_length_dependency('../Data/Multi/updated_LJ/ref1/stretch_15000_folder/job2/system_drag_Ff.txt')
    # obj = drag_length_dependency('../Data/Multi/contact_area_cut110/cut110/stretch_15000_folder/job2/system_drag_Ff.txt')
    
    
    # vel_compare.pop(4)
    # vel_compare.pop(0)
    # obj = drag_length_compare(custom_comp)
    # dt_dependency(dt_files, dt_vals, drag_cap = 100)

    plt.show()
    