from analysis_utils import *

def drag_length_dependency(filename):
    data = analyse_friction_file(filename)    
    time = data['time']
    COM = data['COM_sheet'][:,0]
    contact = data['contact'][0]
    
    quantile = 0.999
    window_len_pct = 0.25
    
        
    group_name = {0: 'full_sheet', 1: 'sheet', 2: 'PB'}
    for g in reversed(range(1)):
        fignum = 0
        if fignum in plt.get_fignums():
            fignum = plt.get_fignums()[-1] + 1
            
        fig = plt.figure(num = fignum)
        fig.suptitle(filename)
        grid = (1,2)
        ax1 = plt.subplot2grid(grid, (0, 0), colspan=1)
        ax2 = plt.subplot2grid(grid, (0, 1), colspan=1)
        
        Ff = data[f'Ff_{group_name[g]}'][:,0]
        # Ff = data['move_force'][:,0]
        
        
        # A = data[f'Ff_{group_name[g]}'][:,0]
        # B = data[f'Ff_{group_name[g]}'][:,1]
        # Ff = np.sqrt(A**2 + B**2)
        
        ax1.plot(time, Ff, label = f'Ydata ({group_name[g]})')
        ax1.plot(time, cum_mean(Ff), label = "Cum mean")
        ax1.plot(time, running_mean(Ff, window_len = int(window_len_pct*len(Ff))), label = "running mean")
        ax1.plot(time, cum_max(Ff), label = "Cum max")
        ax1.plot(time, cumTopQuantileMax(Ff, quantile, brute_force = False), label = f"Top {quantile*100}% max mean")
        ax1.set(xlabel="Time", ylabel='$F_\parallel$ [eV/Å]')
        ax1.legend(loc = 'lower center', fontsize = 10, ncol=2, fancybox = True, shadow = True)
              
        ax2.plot(time, contact, label = "Ydata (full sheet)")
        ax2.plot(time, cum_mean(contact), label = "Cum mean")
        ax2.plot(time, running_mean(contact, window_len = int(window_len_pct*len(contact))), label = "running mean")
        ax2.legend(loc = 'lower center', fontsize = 10, ncol=2, fancybox = True, shadow = True)
        ax2.set(xlabel="Time", ylabel='Contact (Full sheet) [%]')
            
        
    for ax in [ax1, ax2]:
        add_xaxis(ax, time, COM, xlabel='COM$\parallel$ [Å]', decimals = 1)    
    
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    return interactive_plotter(fig)
    
    
def drag_length_compare(filenames):
    group_name = {0: 'full_sheet', 1: 'sheet', 2: 'PB'}
    
    xaxis = "COM" # 'time' || 'COM'
    relative = False
    
    g = 0
        
    quantile = 0.999
    window_len_pct = 0.5
    
    fig1 = plt.figure(num = 0)
    fig1.suptitle("Mean $F_\parallel$ [eV/Å]")
    
    grid = (2,2)
    ax1 = plt.subplot2grid(grid, (0, 0), colspan=1)
    ax2 = plt.subplot2grid(grid, (0, 1), colspan=1)
    ax3 = plt.subplot2grid(grid, (1, 0), colspan=1)
    ax4 = plt.subplot2grid(grid, (1, 1), colspan=1)
    
    
    fig2 = plt.figure(num = 1)
    fig2.suptitle("Max $F_\parallel$ [eV/Å]")
    grid = (2,2)
    ax5 = plt.subplot2grid(grid, (0, 0), colspan=1)
    ax6 = plt.subplot2grid(grid, (0, 1), colspan=1)
    ax7 = plt.subplot2grid(grid, (1, 0), colspan=1)
    ax8 = plt.subplot2grid(grid, (1, 1), colspan=1)
    
    fig3 = plt.figure(num = 2)
    fig3.suptitle("Mean contact [%]")
    grid = (2,2)
    ax9 = plt.subplot2grid(grid, (0, 0), colspan=1)
    ax10 = plt.subplot2grid(grid, (0, 1), colspan=1)
    ax11 = plt.subplot2grid(grid, (1, 0), colspan=1)
    ax12 = plt.subplot2grid(grid, (1, 1), colspan=1)
    
    for filename in filenames:
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
        cummean = cum_mean(Ff)
        runmean = running_mean(Ff, window_len = int(window_len_pct*len(Ff)))
        
        
        # Mean
        output = np.flip(cum_std(np.flip(cummean), step = 5000))
        map = ~np.isnan(output)
        ax1.plot(x, cummean)
        ax2.plot(x[~np.isnan(output)], output[~np.isnan(output)], "-o", markersize = 3, label = filename)
        
         # Running mean 
        output = np.flip(cum_std(np.flip(runmean), step = 5000))
        map = ~np.isnan(output)
        ax3.plot(x, runmean)
        ax4.plot(x[~np.isnan(output)], output[~np.isnan(output)], "-o", markersize = 3)
        ax3.set_xlim(ax1.get_xlim())
        ax4.set_xlim(ax2.get_xlim())
    
        
        # --- Ff (max) --- # 
        cummean_topmax = cumTopQuantileMax(np.abs(data[f'Ff_{group_name[g]}'][:,0]), quantile)
        cummax = cum_max(data[f'Ff_{group_name[g]}'][:,0])
       
        # Mean top quantile max
        output = np.flip(cum_std(np.flip(cummean_topmax), step = 5000))
        map = ~np.isnan(output)
        ax5.plot(x, cummean_topmax)
        ax6.plot(x[~np.isnan(output)], output[~np.isnan(output)], "-o", markersize = 3, label = filename)
       
        # Max 
        output = np.flip(cum_std(np.flip(cummax), step = 5000))
        map = ~np.isnan(output)
        ax7.plot(x, cummax)
        ax8.plot(x[~np.isnan(output)], output[~np.isnan(output)], "-o", markersize = 3)
    
        # --- Contact --- #
        cummean = cum_mean(contact)
        runmean = running_mean(contact, window_len = int(window_len_pct*len(contact)))
        
        # Mean 
        output = np.flip(cum_std(np.flip(cummean), step = 5000))
        map = ~np.isnan(output)
        ax9.plot(x, cummean, label = filename)
        ax10.plot(x[~np.isnan(output)], output[~np.isnan(output)], "-o", markersize = 3)
        
        # Running mean 
        output = np.flip(cum_std(np.flip(runmean), step = 5000))
        map = ~np.isnan(output)
        ax11.plot(x, runmean)
        ax12.plot(x[~np.isnan(output)], output[~np.isnan(output)], "-o", markersize = 3)
        ax11.set_xlim(ax9.get_xlim())
        ax12.set_xlim(ax10.get_xlim())
    
    ax1.set(xlabel=xlabel, ylabel='cum mean')
    ax2.set(xlabel=xlabel, ylabel='reverse cum std')
    ax3.set(xlabel=xlabel, ylabel='running mean')
    ax4.set(xlabel=xlabel, ylabel='reverse cum std')
    
    ax5.set(xlabel=xlabel, ylabel='cum mean top max')
    ax6.set(xlabel=xlabel, ylabel='reverse cum std')
    ax7.set(xlabel=xlabel, ylabel='cum max')
    ax8.set(xlabel=xlabel, ylabel='reverse cum std')
    
    ax9.set(xlabel=xlabel, ylabel='cum mean')
    ax10.set(xlabel=xlabel, ylabel='reverse cum std')
    ax11.set(xlabel=xlabel, ylabel='running mean')
    ax12.set(xlabel=xlabel, ylabel='reverse cum std')
    
    # for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]:
    #     add_xaxis(ax, time, COM, xlabel='COM$\parallel$ [Å]', decimals = 1)    
    
    obj = []
    for fig in [fig1, fig2, fig3]:  
        fig.legend(loc = 'lower center', fontsize = 10, ncol=1, fancybox = True, shadow = True)
        fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
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
    PF = "drag_length" 
    # PF = "drag_length_200nN" 
    # PF = "drag_length_s200nN" 
    
    ref = f'../Data/Baseline/{PF}/ref/system_ref_Ff.txt'
    
    v05 = f'../Data/Baseline/{PF}/v05/system_v05_Ff.txt'
    v5 = f'../Data/Baseline/{PF}/v5/system_v5_Ff.txt'
    v10 = f'../Data/Baseline/{PF}/v10/system_v10_Ff.txt'
    v20 = f'../Data/Baseline/{PF}/v20/system_v20_Ff.txt'
    v50 = f'../Data/Baseline/{PF}/v50/system_v50_Ff.txt'
    v100 = f'../Data/Baseline/{PF}/v100/system_v100_Ff.txt'
    
    K0 = f'../Data/Baseline/{PF}/K0/system_K0_Ff.txt'
    
    T5 = f'../Data/Baseline/{PF}/T5/system_T5_Ff.txt'
    T300 = f'../Data/Baseline/{PF}/T300/system_T300_Ff.txt'
   
    amorph = f'../Data/Baseline/{PF}/amorph/system_amorph_Ff.txt'
    gold = f'../Data/Baseline/{PF}/gold/system_gold_Ff.txt'
   
    vel_compare = [v05, ref, v5, v10, v20, v50, v100]
    temp_compare = [T5, ref, T300]
    K_compare = [ref, K0]
    substrate_compare = [ref, amorph, gold]
    
    v10_comp = [f'../Data/Baseline/{PF}/v10/system_v10_Ff.txt' for PF in ['drag_length', 
                                                                             'drag_length_200nN', 
                                                                             'drag_length_s200nN']]
    # vel_compare.pop(0)
    
    obj = drag_length_dependency('../Data/Baseline/drag_length_s200nN/v10/system_v10_Ff.txt')
    
    
    # obj = drag_length_compare(v10_comp)
    # dt_dependency(dt_files, dt_vals, drag_cap = 100)
    
    plt.show()
    