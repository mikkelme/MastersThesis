from analysis_utils import *

def drag_length_dependency(filename):
    data = analyse_friction_file(filename)    
    time = data['time']
    COM = data['COM_sheet'][:,0]
    contact = data['contact'][0]
    
    quantile = 0.999
    window_len_pct = 0.5
    
    
    
    group_name = {0: 'full_sheet', 1: 'sheet', 2: 'PB'}
    for g in reversed(range(1)):
        fig = plt.figure(num = 100 + g)
        fig.suptitle(filename)
        grid = (1,2)
        ax1 = plt.subplot2grid(grid, (0, 0), colspan=1)
        ax2 = plt.subplot2grid(grid, (0, 1), colspan=1)
        
        Ff = data[f'Ff_{group_name[g]}'][:,0]
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
    
    g = 0
    relative = False
        
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
    
    xlabel = 'Time [ps]'
    for filename in filenames:
        data = analyse_friction_file(filename)   
        COM = data['COM_sheet'][:,0]
        time = data['time']
        contact = data['contact'][0]
        Ff = data[f'Ff_{group_name[g]}'][:,0]
        
        if relative:
            xlabel = 'Rel COM$_\parallel$'
            time /= time[-1] # relative drag
        
        
        # --- Ff (mean) --- # 
        cummean = cum_mean(Ff)
        runmean = running_mean(Ff, window_len = int(window_len_pct*len(Ff)))
        
        
        # Mean
        output = np.flip(cum_std(np.flip(cummean), step = 5000))
        map = ~np.isnan(output)
        ax1.plot(time, cummean)
        ax2.plot(time[~np.isnan(output)], output[~np.isnan(output)], "-o", markersize = 3, label = filename)
        
         # Running mean 
        output = np.flip(cum_std(np.flip(runmean), step = 5000))
        map = ~np.isnan(output)
        ax3.plot(time, runmean)
        ax4.plot(time[~np.isnan(output)], output[~np.isnan(output)], "-o", markersize = 3)
        ax3.set_xlim(ax1.get_xlim())
        ax4.set_xlim(ax2.get_xlim())
    
        
        # --- Ff (max) --- # 
        cummean_topmax = cumTopQuantileMax(np.abs(data[f'Ff_{group_name[g]}'][:,0]), quantile)
        cummax = cum_max(data[f'Ff_{group_name[g]}'][:,0])
       
        # Mean top quantile max
        output = np.flip(cum_std(np.flip(cummean_topmax), step = 5000))
        map = ~np.isnan(output)
        ax5.plot(time, cummean_topmax)
        ax6.plot(time[~np.isnan(output)], output[~np.isnan(output)], "-o", markersize = 3, label = filename)
       
        # Max 
        output = np.flip(cum_std(np.flip(cummax), step = 5000))
        map = ~np.isnan(output)
        ax7.plot(time, cummax)
        ax8.plot(time[~np.isnan(output)], output[~np.isnan(output)], "-o", markersize = 3)
    
        # --- Contact --- #
        cummean = cum_mean(contact)
        runmean = running_mean(contact, window_len = int(window_len_pct*len(contact)))
        
        # Mean 
        output = np.flip(cum_std(np.flip(cummean), step = 5000))
        map = ~np.isnan(output)
        ax9.plot(time, cummean, label = filename)
        ax10.plot(time[~np.isnan(output)], output[~np.isnan(output)], "-o", markersize = 3)
        
        # Running mean 
        output = np.flip(cum_std(np.flip(runmean), step = 5000))
        map = ~np.isnan(output)
        ax11.plot(time, runmean)
        ax12.plot(time[~np.isnan(output)], output[~np.isnan(output)], "-o", markersize = 3)
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
    
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]:
        add_xaxis(ax, time, COM, xlabel='COM$\parallel$ [Å]', decimals = 1)    
    
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
    # filename = '../Data/NG4_newpot_long/nocut_nostretch/_nocut_nostretch_Ff.txt'
    # filename = '../Data/BIG_MULTI_nocut/stretch_30974_folder/job9/system_ext_Ff.txt'
    # filename = '../Data/BIG_MULTI_Ydrag/stretch_30974_folder/job9/system_ext_Ff.txt'
    
    filenames = []
    filenames +=  ['../Data/Baseline/drag_length/ref/system_ref_Ff.txt']
    
    # filenames += ['../Data/Baseline/drag_length/v4d/system_v4d_Ff.txt']
    # filenames += ['../Data/Baseline/drag_length/v4u/system_v4u_Ff.txt']
    
    # filenames += ['../Data/Baseline/drag_length/T5/system_T5_Ff.txt']
    # filenames += ['../Data/Baseline/drag_length/T300/system_T300_Ff.txt']
    
    # filenames += ['../Data/Baseline/dt/dt_0.002/system_dt_0.002_Ff.txt']
    # filenames += ['../Data/Baseline/dt/dt_0.0005/system_dt_0.0005_Ff.txt']
    # filenames += ['../Data/Baseline/dt/dt_0.00025/system_dt_0.00025_Ff.txt']
    
    dt_files = ['../Data/Baseline/dt/dt_0.002/system_dt_0.002_Ff.txt', 
               '../Data/Baseline/drag_length/ref/system_ref_Ff.txt',
               '../Data/Baseline/dt/dt_0.0005/system_dt_0.0005_Ff.txt',
               '../Data/Baseline/dt/dt_0.00025/system_dt_0.00025_Ff.txt']
    dt_vals = [0.002, 0.001, 0.0005, 0.00025]
    
    
    
    # compare = ['../Data/Baseline/drag_length/ref_HFN/system_ref_HFN_Ff.txt',
    #            '../Data/Baseline/drag_length/HFN_K0/system_HFN_K0_Ff.txt', 
    #            '../Data/Baseline/drag_length/HFN_T300/system_HFN_T300_Ff.txt']
    
    compare = ['../Data/Baseline/drag_length/ref/system_ref_Ff.txt',
               '../Data/Baseline/drag_length/v4d/system_v4d_Ff.txt', 
               '../Data/Baseline/drag_length/v4u/system_v4u_Ff.txt']
    
    # drag_length_dependency('../Data/BIG_MULTI_nocut/stretch_5000_folder/job8/system_ext_Ff.txt') # MULTI DRAG
    # drag_length_dependency('../Data/BIG_MULTI_Ydrag/stretch_30974_folder/job9/system_ext_Ff.txt') # MULTI DRAG
    
    obj = drag_length_dependency('../Data/Baseline/drag_length/ref_HFN/system_ref_HFN_Ff.txt')
    # obj = drag_length_compare(compare)
    # dt_dependency(dt_files, dt_vals, drag_cap = 100)
    
    plt.show()