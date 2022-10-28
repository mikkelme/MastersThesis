from analysis_utils import *
from rupture_detect import *
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable

def read_info_file(filename):
    stretch_pct, F_N = np.loadtxt(filename, unpack=True, delimiter = ',')
    return stretch_pct, F_N
    

def read_multi_folder(folders):
    info_file = 'info_file.txt'
    friction_ext = 'Ff.txt'
    chist_ext = 'chist.txt'
    eval_rupture = True
    ruptol = 0 # 0.5
    # group = 0 # full_sheet = 0, sheet = 1, PB = 2
    
    
    for folder in folders:
        data = []
        stretchfile = find_single_file(folder, ext = chist_ext)
        # stretchfile = None
        for a, stretch_dir in enumerate(get_dirs_in_path(folder)):
            alen = len(get_dirs_in_path(folder))
            for b, job_dir in enumerate(get_dirs_in_path(stretch_dir)):
                blen = len(get_dirs_in_path(stretch_dir))
                progress = a * blen + b
                total = alen * blen
                print(f"\r ({progress}/{total}) | {job_dir} ", end = " ")
              
                try:
                    stretch_pct, F_N = read_info_file(os.path.join(job_dir,info_file))
                    if eval_rupture:
                        chist_file = find_single_file(job_dir, ext = chist_ext)
                        rupture_score = detect_rupture(chist_file, stretchfile)
                        # print(rupture_score)
                    else: 
                        # rupture_score = random.uniform(0,1)
                        rupture_score = 0
                    
                    ##############################################
                    # Quick fix for missing stretch pct
                    timestep = int(job_dir.split("stretch.")[1].split("_")[0])
                    stretch_pct = (timestep-5000)/(10999-5000)*0.30
                    ##############################################
                    
                    friction_file = find_single_file(job_dir, ext = friction_ext)
                     
                    Ff, FN = get_fricton_force(friction_file)
                    data.append((stretch_pct, F_N, Ff, rupture_score)) 
                except FileNotFoundError:
                    # print(f" --> Missing files in: {job_dir} ")
                    print(f"<-- Missing files")
        print()
        data = np.array(data, dtype = 'object')
        stretch_pct, F_N, Ff, rup = organize_data(data)
        
       
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
      
        
        group_name = {0: 'Full sheet', 1: 'Sheet', 2: 'PB'}
        linewidth = 1.5
        marker = 'o'
        markersize = 2.5
      
        rup_marker = 'x'
        rupmarkersize = markersize * 3
        
        
        for group in reversed(range(3)):
            # --- Plotting --- #
            fig = plt.figure(num = group)
            fig.suptitle(group_name[group])
            grid = (2,2)
            ax1 = plt.subplot2grid(grid, (0, 0), colspan=1)
            ax2 = plt.subplot2grid(grid, (0, 1), colspan=1)
            ax3 = plt.subplot2grid(grid, (1, 0), colspan=1)
            ax4 = plt.subplot2grid(grid, (1, 1), colspan=1)
            cmap = matplotlib.cm.viridis

            for i in range(len(stretch_pct)):
                color = get_color_value(stretch_pct[i], np.min(stretch_pct), np.max(stretch_pct), cmap=cmap)
                rup_true = np.argwhere(rup[i, :] > ruptol)
                rup_false = np.argwhere(rup[i, :] <= ruptol)
                
                ax1.plot(F_N, Ff[i, :, group, 0], color = color, linewidth = linewidth, label = f'stretch = {stretch_pct[i]:g}')
                ax1.plot(F_N[rup_true], Ff[i, rup_true, group, 0], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
                ax1.plot(F_N[rup_false], Ff[i, rup_false, group, 0], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
                
            
                ax2.plot(F_N, Ff[i, :, group, 1], color = color, linewidth = linewidth, label = f'stretch = {stretch_pct[i]:g}')
                ax2.plot(F_N[rup_true], Ff[i, rup_true, group, 1], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
                ax2.plot(F_N[rup_false], Ff[i, rup_false, group, 1], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
                
                
            norm = matplotlib.colors.BoundaryNorm(stretch_pct, cmap.N)
            cax = make_axes_locatable(ax2).append_axes("right", "5%")
            cax.grid(False)
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='Stretch')
            
            
            ax1.set(xlabel='$F_N$', ylabel='max $F_\parallel$')
            ax2.set(xlabel='$F_N$', ylabel='mean $F_\parallel$')
            
            
            for j in range(len(F_N)):                
                color = get_color_value(F_N[j], np.min(F_N), np.max(F_N))
                rup_true = np.argwhere(rup[:, j] > ruptol)
                rup_false = np.argwhere(rup[:, j] <= ruptol)
                
                ax3.plot(stretch_pct, Ff[:, j, group, 0], color = color, linewidth = linewidth, markersize = markersize, label = f'F_N = {F_N[j]:g}')
                ax3.plot(stretch_pct[rup_true], Ff[rup_true, j, group, 0], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
                ax3.plot(stretch_pct[rup_false], Ff[rup_false, j, group, 0], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
                
                ax4.plot(stretch_pct, Ff[:, j, group, 1], color = color, linewidth = linewidth, markersize = markersize, label = f'F_N = {F_N[j]:g}')
                ax4.plot(stretch_pct[rup_true], Ff[rup_true, j, group, 1], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
                ax4.plot(stretch_pct[rup_false], Ff[rup_false, j, group, 1], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
                
                
            norm = matplotlib.colors.BoundaryNorm(F_N, cmap.N)
            cax = make_axes_locatable(ax4).append_axes("right", "5%")
            cax.grid(False)
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='$F_N$')
            
            ax3.set(xlabel='stretch [%]', ylabel='max $F_\parallel$')
            ax4.set(xlabel='stretch [%]', ylabel='mean $F_\parallel$')
 
            plt.tight_layout()       
        # fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)      
        plt.show()
        
        
        
        
        
        # plt.figure(num = 0)
        # for i in range(len(stretch_pct)):
        #     plt.suptitle(group_name[group])
        #     color = get_color_value(stretch_pct[i], np.min(stretch_pct), np.max(stretch_pct))
            
        #     plt.subplot(2,1,1)
        #     plt.plot(F_N, Ff[i, :, group, 0], color = color, linewidth = linewidth, label = f'stretch pct = {stretch_pct[i]:g}')
        #     plt.ylabel("max $F_\parallel$")
            
        #     # Conditional markers for ruptures
        #     rup_true = np.argwhere(rup[i, :] >= ruptol)
        #     rup_false = np.argwhere(rup[i, :] < ruptol)
        #     plt.plot(F_N[rup_true], Ff[i, rup_true, group, 0], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
        #     plt.plot(F_N[rup_false], Ff[i, rup_false, group, 0], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
            
         
            
        #     plt.subplot(2,1,2)
        #     plt.plot(F_N, Ff[i, :, group, 1], color = color, linewidth = linewidth, label = f'stretch pct = {stretch_pct[i]:g}')
        #     plt.ylabel("mean $F_\parallel$")
            
        #     # Conditional markers for ruptures
        #     plt.plot(F_N[rup_true], Ff[i, rup_true, group, 1], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
        #     plt.plot(F_N[rup_false], Ff[i, rup_false, group, 1], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
            
        # plt.xlabel("$F_N$")
        # plt.legend()        
        
        
        # plt.figure(num = 1)
        # for j in range(len(F_N)):
        #     plt.suptitle(group_name[group])
            
        #     color = get_color_value(F_N[j], np.min(F_N), np.max(F_N))
            
        #     plt.subplot(2,1,1)
        #     plt.plot(stretch_pct, Ff[:, j, group, 0], color = color, linewidth = linewidth, markersize = markersize, label = f'F_N = {F_N[j]:g}')
        #     plt.ylabel("max $F_\parallel$")
            
            
        #     # Conditional markers for ruptures
        #     rup_true = np.argwhere(rup[:, j] >= ruptol)
        #     rup_false = np.argwhere(rup[:, j] < ruptol)
        #     plt.plot(stretch_pct[rup_true], Ff[rup_true, j, group, 0], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
        #     plt.plot(stretch_pct[rup_false], Ff[rup_false, j, group, 0], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
            
        #     plt.subplot(2,1,2)
        #     plt.plot(stretch_pct, Ff[:, j, group, 1], color = color, linewidth = linewidth, markersize = markersize, label = f'F_N = {F_N[j]:g}')
        #     plt.ylabel("mean $F_\parallel$")
        #     plt.plot(stretch_pct[rup_true], Ff[rup_true, j, group, 1], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
        #     plt.plot(stretch_pct[rup_false], Ff[rup_false, j, group, 1], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
            
            
        # plt.xlabel("stretch [%]")
        # plt.legend()
            
        
            
        plt.show()
        
        
     


if __name__ == "__main__":
    # folders = ['../Data/one_config_multi_data']
    folders = ['../Data/OCMD_newpot']
    read_multi_folder(folders)