from analysis_utils import *

def read_info_file(filename):
    stretch_pct, F_N = np.loadtxt(filename, unpack=True, delimiter = ',')
    return stretch_pct, F_N
    

def read_multi_folder(folders):
    for folder in folders:
        data = []
        num_stretch = 0
        for stretch_dir in get_dirs_in_path(folder):
            num_stretch += 1
            for job_dir in get_dirs_in_path(stretch_dir):
                stretch_pct, F_N = read_info_file(os.path.join(job_dir,'info_file.txt'))
                try:
                    Ff, FN = get_fricton_force(os.path.join(job_dir,'friction_force_tmp.txt'))
                except:
                    print(os.path.join(job_dir,'friction_force_tmp.txt'))
                    pass
                data.append((stretch_pct, F_N, Ff))
       
        data = np.array(data, dtype = 'object')
        stretch_pct, F_N, Ff = organize_data(data)
        
        
        group = 0 # full_sheet = 0, sheet = 1, PB = 2
        group_name = {0: 'Full sheet', 1: 'Sheet', 2: 'PB'}
        linewidth = 1.5
        markersize = 2.5
        
        
        plt.figure(num = 0)
        for i in range(num_stretch):
            plt.suptitle(group_name[group])
            color = get_color_value(stretch_pct[i], np.min(stretch_pct), np.max(stretch_pct))
            
            plt.subplot(2,1,1)
            plt.plot(F_N, Ff[i, :, group, 0], "-o", color = color, linewidth = linewidth, markersize = markersize, label = f'stretch pct = {stretch_pct[i]:g}')
            plt.ylabel("max Ff")
            
            plt.subplot(2,1,2)
            plt.plot(F_N, Ff[i, :, group, 1], "-o", color = color,  linewidth = linewidth, markersize = markersize, label = f'stretch pct = {stretch_pct[i]:g}')
            plt.ylabel("mean Ff")
            
        plt.xlabel("F_N")
        plt.legend()        
        
        
        plt.figure(num = 1)
        for j in range(len(F_N)):
            plt.suptitle(group_name[group])
            
            color = get_color_value(F_N[j], np.min(F_N), np.max(F_N))
            
            plt.subplot(2,1,1)
            plt.plot(stretch_pct, Ff[:, j, group, 0], "-o", color = color, linewidth = linewidth, markersize = markersize, label = f'F_N = {F_N[j]:g}')
            plt.ylabel("max Ff")
            
            plt.subplot(2,1,2)
            plt.plot(stretch_pct, Ff[:, j, group, 1], "-o", color = color, linewidth = linewidth, markersize = markersize, label = f'F_N = {F_N[j]:g}')
            plt.ylabel("mean Ff")
            
        plt.xlabel("stretch pct")
        plt.legend()
            
        
            
        plt.show()
        
        
     


if __name__ == "__main__":
    folders = ['../Data/one_config_multi_data']
    # folders = ['../Data/one_config_multi_data_small']
    read_multi_folder(folders)