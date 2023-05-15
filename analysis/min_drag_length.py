### Scripts for the analysis of the effect
### from sliding length (drag length)


from analysis_utils import *

def min_drag_length(filenames, start = 50, rel_std_lim = 0.1, step = 10):
    g = 0
    
    mean_pct = 0.5
    std_pct = 0.2 * mean_pct
    
    group_name = {0: 'full_sheet', 1: 'sheet', 2: 'PB'}
    
    
    stretch_pct = np.empty(len(filenames))
    F_N = np.empty(len(filenames))
    Ff_min_drag = np.zeros(len(filenames))
    contact_min_drag = np.zeros(len(filenames))
    for i, filename in enumerate(filenames):
        print(f"\r ({i+1}/{len(filenames)}) | {filename} ", end = " ")
        
        data = analyse_friction_file(filename)    
        info = read_info_file('/'.join(filename.split('/')[:-1]) + '/info_file.txt' )
        
        stretch_pct[i] = info['stretch_max_pct']
        F_N[i] = metal_to_SI(info['F_N'], 'F')*1e9

        
        Ff = data[f'Ff_{group_name[g]}'][:,0]
        contact = data['contact'][:,0]
        time = data['time']
        VA_pos = (time - time[0]) * info['drag_speed']  # virtual atom position
    
        Ff_min_drag[i] = std_under_lim(Ff, VA_pos, start, step, rel_std_lim, mean_pct, std_pct)
        contact_min_drag[i] = std_under_lim(contact, VA_pos, start, step, rel_std_lim, mean_pct, std_pct)
        
    print()   
    return stretch_pct, F_N, Ff_min_drag, contact_min_drag

    
def std_under_lim(arr, VA_pos, start, step, rel_std_lim, mean_pct, std_pct):
    for pos in range(start, int(VA_pos[-1])+1, step):
        arg = np.argmin(np.abs(VA_pos - pos))
        
        mean_window = int(mean_pct * arg)
        std_window = int(std_pct * arg)
        
        mean, std = mean_cut_and_std(arr[:arg], mean_window, std_window)
        std_rel = std / mean
        
        if std_rel <= rel_std_lim:
            return VA_pos[arg]
        
    return VA_pos[-1]
            


def stability_folder(folder, start = 50, rel_std_lim = 0.01, step = 10):
    filenames = get_files_in_folder(folder, ext = '_Ff.txt')
    stretch_pct, F_N, Ff_min_drag, contact_min_drag = min_drag_length(filenames, start, rel_std_lim, step)
    data = np.array([stretch_pct, F_N, Ff_min_drag, contact_min_drag], dtype = 'object').T
    stretch_pct, F_N, Ff_min_drag, contact_min_drag = organize_data(data)
    
    plot_heatmap(Ff_min_drag, ("stretch", stretch_pct), ("$F_N$", F_N), "Ff")
    plot_heatmap(contact_min_drag, ("stretch", stretch_pct), ("$F_N$", F_N), "Contact")
     
    

    

if __name__ == "__main__":
   
    filenames = ['../Data/Multi/cuts/ref2/stretch_230000_folder/job4/system_drag_Ff.txt']
    # min_drag_length(filenames)
    
    stability_folder("../Data/Multi/cuts/ref4", start = 50, rel_std_lim = 0.1, step = 20)
    # vel_compare.pop(4)
    # vel_compare.pop(0)
    # obj = drag_length_compare(custom_comp)
    # dt_dependency(dt_files, dt_vals, drag_cap = 100)

    plt.show()
    