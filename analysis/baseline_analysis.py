from analysis_utils import *

def drag_length_dependency(filename):
    data = analyse_friction_file(filename)    
    time = data['time']
    COM = data['COM_sheet']
    
    
    group_name = {0: 'full_sheet', 1: 'sheet', 2: 'PB'}
    for g in reversed(range(3)):
        plt.figure(num = g) # mean
        plt.title(filename)
        plt.plot(COM[:,0], data[f'Ff_{group_name[g]}'][:,0], label = group_name[g])
        plt.plot(COM[:,0], cum_mean(data[f'Ff_{group_name[g]}'][:,0]), label = "Cum mean")
        plt.plot(COM[:,0], cum_max(data[f'Ff_{group_name[g]}'][:,0]), label = "Cum max")
        plt.plot(COM[:,0], cum_mean(np.abs(data[f'Ff_{group_name[g]}'][:,0])), label = "Cum mean abs")
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
    
    
    
    # drag_length_dependency(filenames[1])
    obj = drag_length_compare(filenames)
    plt.show()