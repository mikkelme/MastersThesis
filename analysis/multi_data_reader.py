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
                Ff, FN = get_fricton_force(os.path.join(job_dir,'friction_force_tmp.txt'))
                data.append((stretch_pct, F_N, Ff))
        # print(subdirs)
        # data = np.asarray(data)
        # data = np.stack(data)
        # print(data[0])
        # print(data)
        data = np.array(data, dtype = 'object')
        # data = np.stack(data)
        # print(data)
        # print(np.shape(data))
        # exit()
        # print(data)
    
        stretch_pct, F_N, Ff = organize_data(data, num_stretch)
        
        
        group = 2 # full_sheet = 0, sheet = 1, PB = 2
        group_name = {0: 'Full sheet', 1: 'Sheet', 2: 'PB'}
        plt.figure(num = 0)
        for i in range(num_stretch):
            plt.suptitle(group_name[group])
            
            plt.subplot(2,1,1)
            plt.title("Max")
            plt.plot(F_N, Ff[i, :, group, 0], "-o", label = f'stretch pct = {stretch_pct[i]:g}')
            
            plt.subplot(2,1,2)
            plt.title("Mean")
            plt.plot(F_N, Ff[i, :, group, 1], "-o", label = f'stretch pct = {stretch_pct[i]:g}')
            
        plt.xlabel("F_N")
        plt.legend()        
        
        
        plt.figure(num = 1)
        for j in range(len(F_N)):
            plt.suptitle(group_name[group])
            
            plt.subplot(2,1,1)
            plt.title("Max")
            plt.plot(stretch_pct, Ff[:, j, group, 0], "-o", label = f'F_N = {F_N[j]:g}')
            plt.ylabel("max Ff")
            
            plt.subplot(2,1,2)
            plt.title("Mean")
            plt.plot(stretch_pct, Ff[:, j, group, 1], "-o", label = f'F_N = {F_N[j]:g}')
            plt.ylabel("mean Ff")
            
        plt.xlabel("stretch pct")
        plt.legend()
            
        # plt.figure(num = 0)
        # for i in range(num_stretch):
        #     plt.plot(F_N, Ff[i], "-o", label = f'stretch pct = {stretch_pct[i]:g}')
        # plt.xlabel("F_N")
        # plt.legend()        
        
        
        # plt.figure(num = 1)
        # for j in range(len(F_N)):
        #     plt.plot(stretch_pct, Ff[:, j], "-o", label = f'F_N = {F_N[j]:g}')
        # plt.xlabel("stretch pct")
        # plt.legend()
            
            
        plt.show()
        
        
       
        # print(F_N/5)
        
        # print(data)
        exit()



if __name__ == "__main__":
    folders = ['../Data/one_config_multi_data']
    read_multi_folder(folders)