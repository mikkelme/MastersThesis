from analysis_utils import *

def drag_length_dependency(filename):
    data = analyse_friction_file(filename)    
    time = data['time']
    COM = data['COM_sheet']
    
    quantile = 0.99
    
    group_name = {0: 'full_sheet', 1: 'sheet', 2: 'PB'}
    for g in reversed(range(3)):
        plt.figure(num = g) # mean
        plt.title(filename)
        Ff = data[f'Ff_{group_name[g]}'][:,0]
        
        
        # plt.plot(COM[:,0], Ff, label = group_name[g])
        # plt.plot(COM[:,0], cum_mean(Ff), label = "Cum mean")
        # plt.plot(COM[:,0], cum_max(Ff), label = "Cum max")
        # plt.plot(COM[:,0], cum_mean(TopQuantileMax(Ff, quantile, mean = False)[1]), label = f"Top {quantile*100}% max mean")
       
        # out = cumTopQuantileMax(Ff, quantile, slow = True) 
        
        Ff = Ff[:2000]
        out1 = cumTopQuantileMax(Ff, quantile, slow = True) 
        out2 = cumTopQuantileMax(Ff, quantile, slow = False) 
        plt.plot(out1, label = "out1")
        plt.plot(out2, label = "out2")
        plt.legend()
        plt.show()
        exit()
        # plt.plot(COM[:,0], cum_mean(np.abs(data[f'Ff_{group_name[g]}'][:,0])), label = "Cum mean abs")
        plt.xlabel('COM$\parallel$ [Å]')
        plt.ylabel('$F_\parallel$ [eV/Å]')
        plt.legend()
    
    # for g in range(3):
    #     plt.figure(num = g+3) # mean
    #     plt.plot(time, data[f'Ff_{group_name[g]}'][:,0], label = group_name[g])
    #     plt.plot(time, cum_mean(data[f'Ff_{group_name[g]}'][:,0]), label = "Cum mean")
    #     plt.plot(time, cum_max(data[f'Ff_{group_name[g]}'][:,0]), label = "Cum max")
    #     plt.plot(time, cum_mean(np.abs(data[f'Ff_{group_name[g]}'][:,0])), label = "Cum mean abs")
    #     plt.xlabel('time [ps]')
    #     plt.ylabel('$F_\parallel$ [eV/Å]')
    #     plt.legend()
    
    # plt.show()
    
    
def drag_length_compare(filenames):
    group_name = {0: 'full_sheet', 1: 'sheet', 2: 'PB'}
    
    g = 0
    relative = False
    
    
    fig = plt.figure(num = 100 + g)
    # fig.suptitle(group_name[group])
    grid = (3,2)
    ax1 = plt.subplot2grid(grid, (0, 0), colspan=1)
    ax2 = plt.subplot2grid(grid, (0, 1), colspan=1)
    ax3 = plt.subplot2grid(grid, (1, 0), colspan=1)
    ax4 = plt.subplot2grid(grid, (1, 1), colspan=1)
    ax5 = plt.subplot2grid(grid, (2, 0), colspan=1)
    ax6 = plt.subplot2grid(grid, (2, 1), colspan=1)
    obj = interactive_plotter(fig)
    
    xlabel = 'COM$_\parallel$'
    for filename in filenames:
        data = analyse_friction_file(filename)   
        x = data['COM_sheet']
        if relative:
            xlabel = 'Rel COM$_\parallel$'
            x /= x[-1] # relative drag
        
        cummean = cum_mean(data[f'Ff_{group_name[g]}'][:,0])
        cummean_abs = cum_mean(np.abs(data[f'Ff_{group_name[g]}'][:,0]))
        cummax = cum_max(data[f'Ff_{group_name[g]}'][:,0])
        
        # Mean
        output = np.flip(cum_std(np.flip(cummean), step = 5000))
        map = ~np.isnan(output)
        ax1.plot(x[:,0], cummean)
        ax2.plot(x[~np.isnan(output),0], output[~np.isnan(output)], "-o", markersize = 3, label = filename)
        ax1.set(xlabel=xlabel, ylabel='cum mean $F_\parallel$ [eV/Å]')
        ax2.set(xlabel=xlabel, ylabel='reverse cum std ')
        
       
        # Mean abs
        output = np.flip(cum_std(np.flip(cummean_abs), step = 5000))
        map = ~np.isnan(output)
        ax3.plot(x[:,0], cummean_abs)
        ax4.plot(x[~np.isnan(output),0], output[~np.isnan(output)], "-o", markersize = 3)
        ax3.set(xlabel=xlabel, ylabel='cum mean abs $F_\parallel$ [eV/Å]')
        ax4.set(xlabel=xlabel, ylabel='reverse cum std ')
       
        # Max 
        output = np.flip(cum_std(np.flip(cummax), step = 5000))
        map = ~np.isnan(output)
        ax5.plot(x[:,0], cummax)
        ax6.plot(x[~np.isnan(output),0], output[~np.isnan(output)], "-o", markersize = 3)
        ax5.set(xlabel=xlabel, ylabel='cum max $F_\parallel$ [eV/Å]')
        ax6.set(xlabel=xlabel, ylabel='reverse cum std ')
 
    fig.legend(loc = 'lower center', fontsize = 10, ncol=2, fancybox = True, shadow = True)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)

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
    
    quantile = 0.999
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
    
    filenames += ['../Data/Baseline/dt/dt_0.002/system_dt_0.002_Ff.txt']
    filenames += ['../Data/Baseline/dt/dt_0.0005/system_dt_0.0005_Ff.txt']
    filenames += ['../Data/Baseline/dt/dt_0.00025/system_dt_0.00025_Ff.txt']
    
    dt_files = ['../Data/Baseline/dt/dt_0.002/system_dt_0.002_Ff.txt', 
               '../Data/Baseline/drag_length/ref/system_ref_Ff.txt',
               '../Data/Baseline/dt/dt_0.0005/system_dt_0.0005_Ff.txt',
               '../Data/Baseline/dt/dt_0.00025/system_dt_0.00025_Ff.txt']
    dt_vals = [0.002, 0.001, 0.0005, 0.00025]
    
    
    
    drag_length_dependency('../Data/Baseline/drag_length/ref/system_ref_Ff.txt')
    # obj = drag_length_compare(filenames)
    # dt_dependency(dt_files, dt_vals, drag_cap = 100)
    
    plt.show()