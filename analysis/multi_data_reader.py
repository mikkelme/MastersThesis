from analysis_utils import *

def read_info_file(filename):
    stretch_pct, F_N = np.loadtxt(filename, unpack=True, delimiter = ',')
    return stretch_pct, F_N
    

def read_multi_folder(folders):
    # TODO: Include full_sheet, sheet, PB for both max and avg
    for folder in folders:
        data = []
        num_stretch = 0
        for stretch_dir in get_dirs_in_path(folder):
            num_stretch += 1
            for job_dir in get_dirs_in_path(stretch_dir):
                stretch_pct, F_N = read_info_file(os.path.join(job_dir,'info_file.txt'))
                Ff_max, _ = get_fricton_force(os.path.join(job_dir,'friction_force_tmp.txt'))
                data.append((stretch_pct, F_N, Ff_max))
        # print(subdirs)
        stretch_pct, F_N, Ff = organize_data(np.array(data), num_stretch)
        
        plt.figure(num = 0)
        for i in range(num_stretch):
            plt.plot(F_N, Ff[i], "-o", label = f'stretch pct = {stretch_pct[i]:g}')
        plt.xlabel("F_N")
        plt.legend()        
        
        plt.figure(num = 1)
        for j in range(len(F_N)):
            plt.plot(stretch_pct, Ff[:, j], "-o", label = f'F_N = {F_N[j]:g}')
        plt.xlabel("stretch pct")
        plt.legend()
            
            
        plt.show()
        
        
       
        # print(F_N/5)
        
        # print(data)
        exit()



if __name__ == "__main__":
    folders = ['../Data/one_config_multi_data']
    read_multi_folder(folders)