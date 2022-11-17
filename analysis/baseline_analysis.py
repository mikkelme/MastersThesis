from analysis_utils import *

def drag_length_dependency(filename):
    data = analyse_friction_file(filename)    
    time = data['time']
    COM = data['COM_sheet']
    
    group_name = {0: 'full_sheet', 1: 'sheet', 2: 'PB'}
    for g in reversed(range(3)):
        plt.figure(num = g) # mean
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
    
    plt.show()
if __name__ == "__main__":
    # filename = '../Data/NG4_newpot_long/nocut_nostretch/_nocut_nostretch_Ff.txt'
    # filename = '../Data/BIG_MULTI_nocut/stretch_30974_folder/job9/system_ext_Ff.txt'
    # filename = '../Data/BIG_MULTI_Ydrag/stretch_30974_folder/job9/system_ext_Ff.txt'
    filename = '../Data/Baseline/drag_length/ref/system_ref_Ff.txt'
    drag_length_dependency(filename)