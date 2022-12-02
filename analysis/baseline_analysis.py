from analysis_utils import *

def drag_length_dependency(filename):
    data = analyse_friction_file(filename)    
    time = data['time']
    COM = data['COM_sheet'][:,0]
    contact = data['contact'][:,0]
    
    info = read_info_file('/'.join(filename.split('/')[:-1]) + '/info_file.txt' )
    VA_pos = (time - time[0]) * info['drag_speed']  # virtual atom position
    
   
    # mean_window = int(np.argmin(np.abs(VA_pos - 50)))
    mean_window = int(0.5*len(time))
    std_window = int(0.2*mean_window)
    
    
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
        ax1.set(xlabel='COM$\parallel$ [Å]', ylabel='$F_\parallel$ [nN]')
        ax1.legend(loc = 'lower center', fontsize = 10, ncol=2, fancybox = True, shadow = True)
              
        ax3.plot(VA_pos, rel_std, label = "Ff std")
        ax3.set(xlabel='COM$\parallel$ [Å]', ylabel='Ff rel. runmean std')
        runmean = running_mean(contact, window_len = mean_window)[0]
        rel_std = running_mean(runmean, window_len = std_window)[1]/runmean[~np.isnan(runmean)][-1]
        
        
        ax2.plot(VA_pos, contact, label = "Ydata (full sheet)")
        ax2.plot(VA_pos, cum_mean(contact), label = "Cum mean")
        ax2.plot(VA_pos, runmean, label = "running mean")
        ax2.legend(loc = 'lower center', fontsize = 10, ncol=2, fancybox = True, shadow = True)
        ax2.set(xlabel='COM$\parallel$ [Å]', ylabel='Contact (Full sheet) [%]')
        
        ref = runmean[~np.isnan(runmean)][-1]
        ax4.plot(VA_pos, rel_std, label = "Contact std")
        ax4.set(xlabel='COM$\parallel$ [Å]', ylabel='Contact rel. runmean std')

        
    # for ax in [ax1, ax2]:
    #     add_xaxis(ax, time, VA_pos, xlabel='COM$\parallel$ [Å]', decimals = 1)    
    
    for fig in [fig1, fig2]:
        fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    return [interactive_plotter(fig) for fig in [fig1, fig2]]
    
    
def drag_length_compare(filenames):
    group_name = {0: 'full_sheet', 1: 'sheet', 2: 'PB'}
    
    xaxis = "COM" # 'time' || 'COM'
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
        data = analyse_friction_file(filename)   
        COM = data['COM_sheet'][:,0]
        time = data['time']
        contact = data['contact'][0]
        Ff = data[f'Ff_{group_name[g]}'][:,0]
        
        
        if xaxis == 'time':
            x = time; xlabel = 'Time [ps]'
        elif xaxis == 'COM':
            x = COM; xlabel = 'COM$_\parallel$ [Å]'
        else:
            print(f'xaxis = {xaxis} is not a known setting.')
        
        if relative:
            xlabel = 'Rel ' + xlabel
            x /= x[-1] # relative drag
        
        
        # --- Ff (mean) --- # 
        print(f"\rFile: ({i+1}/{len(filenames)}) | Mean Ff | {filename}       ", end = " ")
        cummean = cum_mean(Ff)
        _, cummean_runstd = running_mean(cummean, window_len = int(runstd_wpct*len(cummean)))
        
        runmean, runstd = running_mean(Ff, window_len = int(runmean_wpct*len(Ff)))
        _, runmean_runstd = running_mean(runmean, window_len = int(runstd_wpct*len(runmean)))
        
        
        # Mean
        ax1.plot(x, cummean)
        ax2.plot(x, cummean_runstd, label = filename)
        # output = np.flip(cum_std(np.flip(cummean)))
        # map = ~np.isnan(output)
        # ax1.plot(x, cummean)
        # ax2.plot(x[~np.isnan(output)], output[~np.isnan(output)], "-o", markersize = 3, label = filename)
        
         # Running mean 
        ax3.plot(x, runmean)
        ax4.plot(x, runmean_runstd)
        ax3.set_xlim(ax1.get_xlim())
        ax4.set_xlim(ax2.get_xlim())
    
        
        # --- Ff (max) --- # 
        print(f"\rFile: ({i+1}/{len(filenames)}) | Max Ff | {filename}       ", end = " ")
        cummean_topmax = cumTopQuantileMax(np.abs(data[f'Ff_{group_name[g]}'][:,0]), quantile)
        _, cummean_topmax_runstd = running_mean(cummean_topmax, window_len = int(runstd_wpct*len(cummean_topmax)))
        
        cummax = cum_max(data[f'Ff_{group_name[g]}'][:,0])
        _, cummax_runstd = running_mean(cummax, window_len = int(runstd_wpct*len(cummax)))

       
        # Mean top quantile max
        ax5.plot(x, cummean_topmax)
        ax6.plot(x, cummean_topmax_runstd, label = filename)
        # output = np.flip(cum_std(np.flip(cummean_topmax)))
        # map = ~np.isnan(output)
        # ax5.plot(x, cummean_topmax)
        # ax6.plot(x[~np.isnan(output)], output[~np.isnan(output)], "-o", markersize = 3, label = filename)
       
        # Max 
        ax7.plot(x, cummax)
        ax8.plot(x, cummax_runstd)
        # output = np.flip(cum_std(np.flip(cummax)))
        # map = ~np.isnan(output)
        # ax7.plot(x, cummax)
        # ax8.plot(x[~np.isnan(output)], output[~np.isnan(output)], "-o", markersize = 3)
    
        # --- Contact --- #
        print(f"\rFile: ({i+1}/{len(filenames)}) | Mean contact | {filename}       ", end = " ")
        cummean = cum_mean(contact)
        _, cummean_runstd = running_mean(cummean, window_len = int(runstd_wpct*len(cummean)))
        
        runmean, runstd = running_mean(contact, window_len = int(runmean_wpct*len(contact)))
        _, runmean_runstd = running_mean(runmean, window_len = int(runstd_wpct*len(runmean)))
        
        
        # Mean 
        ax9.plot(x, cummean, label = filename)
        ax10.plot(x, cummean_runstd)
        
        # output = np.flip(cum_std(np.flip(cummean)))
        # map = ~np.isnan(output)
        # ax9.plot(x, cummean, label = filename)
        # ax10.plot(x[~np.isnan(output)], output[~np.isnan(output)], "-o", markersize = 3)
        
        # Running mean 
        ax11.plot(x, runmean)
        ax12.plot(x, runmean_runstd)
        ax11.set_xlim(ax9.get_xlim())
        ax12.set_xlim(ax10.get_xlim())
  
    print()
    
    ax1.set(xlabel=xlabel, ylabel='cum mean')
    ax2.set(xlabel=xlabel, ylabel=f'cum mean run std ({runstd_wpct*100:.0f}%)')
    ax3.set(xlabel=xlabel, ylabel=f'run mean ({runmean_wpct*100:.0f}%)')
    ax4.set(xlabel=xlabel, ylabel=f'run mean run std ({runstd_wpct*100:.0f}%)')
    
    ax5.set(xlabel=xlabel, ylabel='cum mean top max')
    ax6.set(xlabel=xlabel, ylabel=f'cum mean top max run std ({runstd_wpct*100:.0f}%)')
    ax7.set(xlabel=xlabel, ylabel='cum max')
    ax8.set(xlabel=xlabel, ylabel=f'cum mean top max run std ({runstd_wpct*100:.0f}%)')
    
    ax9.set(xlabel=xlabel, ylabel='cum mean')
    ax10.set(xlabel=xlabel, ylabel=f'cum mean run std ({runstd_wpct*100:.0f}%)')
    ax11.set(xlabel=xlabel, ylabel=f'run mean ({runmean_wpct*100:.0f}%)')
    ax12.set(xlabel=xlabel, ylabel=f'run mean run std ({runstd_wpct*100:.0f}%)')
    
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


def dt_dependency(filenames, dt, drag_cap = 0):
    fig = plt.figure()
    grid = (3,1)
    ax1 = plt.subplot2grid(grid, (0, 0), colspan=1)
    ax2 = plt.subplot2grid(grid, (1, 0), colspan=1)
    ax3 = plt.subplot2grid(grid, (2, 0), colspan=1)
    
    Ffmax = np.zeros(len(filenames))
    Ffmean = np.zeros(len(filenames))
    Fftopmax= np.zeros(len(filenames))
    dt = np.array(dt)
    
    quantile = 0.99
    for i, filename in enumerate(filenames):
        data = analyse_friction_file(filename)  
        
        if drag_cap > 0:
            COM = data['COM_sheet'][:,0]
            Ff = data['Ff_full_sheet'][COM <= drag_cap, 0]
            Ffmax[i] = np.max(Ff)
            Ffmean[i] = np.mean(Ff)
            
            topN, Fftopmax[i] = TopQuantileMax(Ff, quantile, mean = True)
            
            # topn = int((1-quantile)*len(Ff))
            # Fftopmax[i] = np.mean(Ff[np.argpartition(Ff, -topn)[-topn:]])
            # print(topn, Fftopmax[i])
            # exit()
            
           
        else:
            Ffmax[i] = data['Ff'][0, 0]
            Ffmean[i] = data['Ff'][0, 1]
        
     
    ax1.plot(dt, Ffmax, linestyle = None, marker = 'o', color = color_cycle(0))
    ax1.set(xlabel='dt $[ps]$', ylabel='max $F_\parallel$ $[eV/Å]$')
    
    ax2.plot(dt, Fftopmax, linestyle = None, marker = 'o', color = color_cycle(1))
    ax2.set(xlabel='dt $[ps]$', ylabel=f'max (top {quantile*100}%) $F_\parallel$ $[eV/Å]$')
    
    
    ax3.plot(dt, Ffmean, linestyle = None, marker = 'o', color = color_cycle(2))
    ax3.set(xlabel='dt $[ps]$', ylabel='mean $F_\parallel$ $[eV/Å]$')
    
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)

    


if __name__ == "__main__":
    # dt_files = ['../Data/Baseline/dt/dt_0.002/system_dt_0.002_Ff.txt', 
    #            '../Data/Baseline/drag_length/ref/system_ref_Ff.txt',
    #            '../Data/Baseline/dt/dt_0.0005/system_dt_0.0005_Ff.txt',
    #            '../Data/Baseline/dt/dt_0.00025/system_dt_0.00025_Ff.txt']
    # dt_vals = [0.002, 0.001, 0.0005, 0.00025]
    
    # Parrent folder
    # PF = "drag_length" 
    PF = "drag_length_200nN" 
    # PF = "drag_length_s200nN" 
    
    ref = f'../Data/Baseline/{PF}/ref/system_ref_Ff.txt'
    
    v05 = f'../Data/Baseline/{PF}/v05/system_v05_Ff.txt'
    v5 = f'../Data/Baseline/{PF}/v5/system_v5_Ff.txt'
    v10 = f'../Data/Baseline/{PF}/v10/system_v10_Ff.txt'
    v20 = f'../Data/Baseline/{PF}/v20/system_v20_Ff.txt'
    v50 = f'../Data/Baseline/{PF}/v50/system_v50_Ff.txt'
    v100 = f'../Data/Baseline/{PF}/v100/system_v100_Ff.txt'
    
    K0 = f'../Data/Baseline/{PF}/K0/system_K0_Ff.txt'
    K5 = f'../Data/Baseline/{PF}/K5/system_K5_Ff.txt'
    K10 = f'../Data/Baseline/{PF}/K10/system_K10_Ff.txt'
    
    T5 = f'../Data/Baseline/{PF}/T5/system_T5_Ff.txt'
    T50 = f'../Data/Baseline/{PF}/T50/system_T50_Ff.txt'
    T200 = f'../Data/Baseline/{PF}/T200/system_T200_Ff.txt'
    T300 = f'../Data/Baseline/{PF}/T300/system_T300_Ff.txt'
   
    amorph = f'../Data/Baseline/{PF}/amorph/system_amorph_Ff.txt'
    gold = f'../Data/Baseline/{PF}/gold/system_gold_Ff.txt'
   
    vel_compare = [v05, ref, v5, v10, v20, v50, v100]
    temp_compare = [T5, ref, T300]
    K_compare = [K5, K10, ref, K0]
    substrate_compare = [ref, amorph, gold]
    
    v10_comp = [f'../Data/Baseline/{PF}/v10/system_v10_Ff.txt' for PF in ['drag_length', 
                                                                             'drag_length_200nN', 
                                                                             'drag_length_s200nN']]
    
    
    
    custom_comp = [ '../Data/Multi/nocuts/ref1/stretch_15000_folder/job0/system_drag_Ff.txt',
                    '../Data/Multi/nocuts/ref1/stretch_15000_folder/job1/system_drag_Ff.txt',
                    '../Data/Multi/nocuts/ref1/stretch_15000_folder/job2/system_drag_Ff.txt']
                #    '../Data/Multi/nocuts/ref1/stretch_315000_folder/job2/system_drag_Ff.txt']
    
    
    obj = drag_length_dependency('../Data/Multi/cuts/ref3/stretch_15000_folder/job0/system_drag_Ff.txt')
    
    
    # vel_compare.pop(4)
    # vel_compare.pop(0)
    # obj = drag_length_compare(custom_comp)
    # dt_dependency(dt_files, dt_vals, drag_cap = 100)

    plt.show()
    