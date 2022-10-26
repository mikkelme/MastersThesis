from analysis_utils import *
from rupture_detect import *
import random

def read_info_file(filename):
    stretch_pct, F_N = np.loadtxt(filename, unpack=True, delimiter = ',')
    return stretch_pct, F_N
    

def read_multi_folder(folders):
    info_file = 'info_file.txt'
    friction_file = '_tmp_Ff.txt'
    chist_file = '_tmp_chist.txt'
    detect_rupture = False
    ruptol = 0.5
    
    
    for folder in folders:
        data = []
        for stretch_dir in get_dirs_in_path(folder):
            for job_dir in get_dirs_in_path(stretch_dir):
              
                try:
                    stretch_pct, F_N = read_info_file(os.path.join(job_dir,info_file))
                    if detect_rupture:
                        rupture_score = detect_rupture(os.path.join(job_dir,chist_file))
                    else: 
                        # rupture_score = random.uniform(0,1)
                        rupture_score = 0
                    
                    ##############################################
                    # Quick fix for missing stretch pct
                    timestep = int(job_dir.split("stretch.")[1].split("_")[0])
                    stretch_pct = (timestep-5000)/(10999-5000)*0.30
                    ##############################################
                    
                    Ff, FN = get_fricton_force(os.path.join(job_dir,friction_file))
                    data.append((stretch_pct, F_N, Ff, rupture_score)) 
                except FileNotFoundError:
                    print(f"Missing files in: {job_dir} ")
       
        data = np.array(data, dtype = 'object')
        stretch_pct, F_N, Ff, rup = organize_data_new(data)
        
       
        detections = [["stretch %", "F_N"]]
        for i in range(len(stretch_pct)):
            for j in range(len(F_N)):
                if rup[i,j] > ruptol:
                    detections.append([stretch_pct[i], F_N[j]])
        
        if len(detections) > 1:
            print("Rupture detected:")
            print(np.array(detections))
        else:
            print("No rupture detected")
      
        
        group = 1 # full_sheet = 0, sheet = 1, PB = 2
        group_name = {0: 'Full sheet', 1: 'Sheet', 2: 'PB'}
        linewidth = 1.5
        marker = 'o'
        markersize = 2.5
      
        rup_marker = 'x'
        rupmarkersize = markersize * 3
        
        plt.figure(num = 0)
        for i in range(len(stretch_pct)):
            plt.suptitle(group_name[group])
            color = get_color_value(stretch_pct[i], np.min(stretch_pct), np.max(stretch_pct))
            
            plt.subplot(2,1,1)
            plt.plot(F_N, Ff[i, :, group, 0], color = color, linewidth = linewidth, label = f'stretch pct = {stretch_pct[i]:g}')
            plt.ylabel("max $F_\parallel$")
            
            # Conditional markers for ruptures
            rup_true = np.argwhere(rup[i, :] >= ruptol)
            rup_false = np.argwhere(rup[i, :] < ruptol)
            plt.plot(F_N[rup_true], Ff[i, rup_true, group, 0], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
            plt.plot(F_N[rup_false], Ff[i, rup_false, group, 0], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
            
         
            
            plt.subplot(2,1,2)
            plt.plot(F_N, Ff[i, :, group, 1], color = color, linewidth = linewidth, label = f'stretch pct = {stretch_pct[i]:g}')
            plt.ylabel("mean $F_\parallel$")
            
            # Conditional markers for ruptures
            plt.plot(F_N[rup_true], Ff[i, rup_true, group, 1], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
            plt.plot(F_N[rup_false], Ff[i, rup_false, group, 1], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
            
        plt.xlabel("$F_N$")
        plt.legend()        
        
        
        plt.figure(num = 1)
        for j in range(len(F_N)):
            plt.suptitle(group_name[group])
            
            color = get_color_value(F_N[j], np.min(F_N), np.max(F_N))
            
            plt.subplot(2,1,1)
            plt.plot(stretch_pct, Ff[:, j, group, 0], color = color, linewidth = linewidth, markersize = markersize, label = f'F_N = {F_N[j]:g}')
            plt.ylabel("max $F_\parallel$")
            
            
            # Conditional markers for ruptures
            rup_true = np.argwhere(rup[:, j] >= ruptol)
            rup_false = np.argwhere(rup[:, j] < ruptol)
            plt.plot(stretch_pct[rup_true], Ff[rup_true, j, group, 0], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
            plt.plot(stretch_pct[rup_false], Ff[rup_false, j, group, 0], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
            
            plt.subplot(2,1,2)
            plt.plot(stretch_pct, Ff[:, j, group, 1], color = color, linewidth = linewidth, markersize = markersize, label = f'F_N = {F_N[j]:g}')
            plt.ylabel("mean $F_\parallel$")
            plt.plot(stretch_pct[rup_true], Ff[rup_true, j, group, 1], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
            plt.plot(stretch_pct[rup_false], Ff[rup_false, j, group, 1], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
            
            
        plt.xlabel("stretch [%]")
        plt.legend()
            
        
            
        plt.show()
        
        
     


if __name__ == "__main__":
    # folders = ['../Data/one_config_multi_data']
    folders = ['../Data/OCMD_newpot']
    read_multi_folder(folders)