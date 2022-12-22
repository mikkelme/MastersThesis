from analysis_utils import *
from rupture_detect import *
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable

from read_stuff import *

def read_info_file_old(filename):
    stretch_pct, F_N = np.loadtxt(filename, unpack=True, delimiter = ',')
    return stretch_pct, F_N
    


def read_multi_folder(folder, mean_pct = 0.5, std_pct = 0.2, stretch_lim = [None, None],  FN_lim = [None, None]):
    # Settings
    info_file = 'info_file.txt'
    friction_ext = 'Ff.txt'
    chist_ext = 'chist.txt'
    
        
    
    data = []
    rupture = []
    # Loop through stretch folders, format: stretch_{TimeStep}_folder
    for i, stretch_folder in enumerate(get_dirs_in_path(folder, sort = True)): 
        num_stretch = len(get_dirs_in_path(folder))
        
        # Loop through F_N folders, format: job{j}
        for j, job_dir in enumerate(get_dirs_in_path(stretch_folder, sort = True)):
            num_FN = len(get_dirs_in_path(stretch_folder))
            progress = i * num_FN + j
            total = num_stretch * num_FN
            print(f"\r ({progress+1}/{total}) | {job_dir} ", end = " ")
            
            try: # If file exist
                # Get run parameters
                info_dict = read_info_file(os.path.join(job_dir,info_file))
                try:
                    is_ruptured = info_dict['is_ruptured']
                except KeyError: # is_ruptred not yet added to file
                    print("Sim not done")
                    continue
                    # is_ruptured = 0
                    
                
                
                stretch_pct = info_dict['SMAX']
                F_N = metal_to_SI(info_dict['F_N'], 'F')*1e9
                
                
                if False:
                    plt.figure(num = unique_fignum())
                    plt.subplot(3,1,1)
                    plt.title(f'{job_dir}\nstretch = {stretch_pct},  F_N = {F_N}')
                    read_vel(os.path.join(job_dir,'vel.txt'), create_fig = False)
                    
                    plt.subplot(3,1,2)
                    read_MSD(os.path.join(job_dir,'MSD.txt'), create_fig = False)
                    # read_CN(os.path.join(job_dir,'CN.txt'), create_fig = False)
                    
                    plt.subplot(3,1,3)
                    read_ystress(os.path.join(job_dir,'YS.txt'), create_fig = False)
                    
                    
                    # plt.title(f'{job_dir}\nstretch = {stretch_pct},  F_N = {F_N}')
                    # dat = read_ave_time(os.path.join(job_dir,'YS.txt'))
                    # runmax = cum_max(dat['c_YS'])
                    # YStol = 0.95*runmax
                    # plt.plot(dat['TimeStep'], dat['c_YS'])
                    # plt.plot(dat['TimeStep'], YStol, linestyle = '--', color = 'black')
                    # plt.ylabel("YS")

                    # plt.subplot(2,1,2)
                    # dat = read_ave_time(os.path.join(job_dir,'CN.txt'))
                    # runmax = cum_max(dat['c_CN_ave'])
                    # CNtol = (1-2/4090)*runmax
                    # plt.plot(dat['TimeStep'], dat['c_CN_ave'])
                    # plt.plot(dat['TimeStep'], CNtol, linestyle = '--', color = 'black')
                    # plt.ylabel("CN")
                    # plt.xlabel("Timestep")
                    
                rupture.append((stretch_pct, F_N, is_ruptured, job_dir))  
                
                if not is_ruptured:
                    # Get data
                    friction_file = find_single_file(job_dir, ext = friction_ext)     
                    fricData = analyse_friction_file(friction_file, mean_pct, std_pct)
                    data.append((stretch_pct, F_N, fricData['Ff'], fricData['Ff_std'], fricData['contact_mean'], fricData['contact_std']))  
            
            except FileNotFoundError:
                print(f"<-- Missing file")
    print()
    
    data = np.array(data, dtype = 'object')
    rupture = np.array(rupture, dtype = 'object')
    stretch_pct, F_N, Ff, Ff_std, contact_mean, contact_std = organize_data(data, stretch_lim, FN_lim)
    rup_stretch_pct, rup_F_N, rup, filenames = organize_data(rupture, stretch_lim, FN_lim)
    
    # --- Rupture detection --- #
    if rup.any():
        # Print information
        detections = [["stretch %", "F_N", "Filenames"]]
        map = np.argwhere(rup == 1)
        for (i,j) in map:
            print(filenames[i,j], folder)
            detections.append([rup_stretch_pct[i], rup_F_N[j], filenames[i,j].removeprefix(folder)])
           
        print(f"{len(detections)-1} Ruptures detected in \'{folder}\':")
        print(np.array(detections))
                
    else:
        print("No rupture detected")
    
    return  (stretch_pct, F_N, Ff, Ff_std, contact_mean, contact_std), (rup_stretch_pct, rup_F_N, rup, filenames)


def plot_multi(folders, mean_pct = 0.5, std_pct = 0.01, stretch_lim = [None, None],  FN_lim = [None, None], show_filename = True):
    
    for folder in folders:
        (stretch_pct, F_N, Ff, Ff_std, contact_mean, contact_std), (rup_stretch_pct, rup_F_N, rup, filenames) = read_multi_folder(folder, mean_pct, std_pct, stretch_lim, FN_lim)
    

        group_name = {0: 'Full sheet', 1: 'Sheet', 2: 'PB'}
        linewidth = 1.5
        marker = 'o'
        markersize = 2.5

        # rup_marker = 'x'
        # rupmarkersize = markersize * 3 

        # --- Plotting --- #
            
        obj_list = []
        for group in reversed(range(3)):
            # --- Max/mean Ff --- #
            fig = plt.figure(num = unique_fignum())
            if show_filename:
                fig.suptitle(group_name[group] + " | " +  folder)
            else:
                fig.suptitle(group_name[group])
            grid = (4,2)
            ax1 = plt.subplot2grid(grid, (0, 0), colspan=1) # (F_N, max Ff)
            ax2 = plt.subplot2grid(grid, (0, 1), colspan=1) # (F_N, mean Ff)
            ax3 = plt.subplot2grid(grid, (1, 0), colspan=1) # (stretch, max Ff)
            ax4 = plt.subplot2grid(grid, (1, 1), colspan=1) # (stretch, mean Ff)
            ax5 = plt.subplot2grid(grid, (2, 0), colspan=1) # (contact(F_N), max Ff)
            ax6 = plt.subplot2grid(grid, (2, 1), colspan=1) # (contact(F_N), mean Ff)
            ax7 = plt.subplot2grid(grid, (3, 0), colspan=1) # (contact(stretch), max Ff)
            ax8 = plt.subplot2grid(grid, (3, 1), colspan=1) # (contact(stretch), mean Ff)
            cmap = matplotlib.cm.viridis

            for i in range(len(stretch_pct)):
                color = get_color_value(stretch_pct[i], np.min(stretch_pct), np.max(stretch_pct), cmap=cmap)
                
                ax1.plot(F_N, Ff[i, :, group, 0], color = color, linewidth = linewidth, label = f'stretch = {stretch_pct[i]:g}')
                ax2.plot(F_N, Ff[i, :, group, 1], color = color, linewidth = linewidth, label = f'stretch = {stretch_pct[i]:g}')
                
                sortidx = np.argsort(contact_mean[i,:,1])
                ax5.plot(contact_mean[i, sortidx,1], Ff[i, sortidx, group, 0], color = color, linewidth = linewidth, markersize = markersize, label = f'stretch = {stretch_pct[i]:g}')                
                ax6.plot(contact_mean[i, sortidx,1], Ff[i, sortidx, group, 1], color = color, linewidth = linewidth, markersize = markersize, label = f'stretch = {stretch_pct[i]:g}')

                
            norm = matplotlib.colors.BoundaryNorm(stretch_pct, cmap.N)
            cax = make_axes_locatable(ax2).append_axes("right", "5%")
            cax.grid(False)
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='Stretch [%]')
            
            cax = make_axes_locatable(ax6).append_axes("right", "5%")
            cax.grid(False)
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='Stretch [%]')
        
        
            ax1.set(xlabel='$F_N$ [nN]', ylabel='max $F_\parallel$ [nN]')
            ax2.set(xlabel='$F_N$ [nN]', ylabel='mean $F_\parallel$ [nN]')
            ax5.set(xlabel='contact ($F_N$) [%]', ylabel='max $F_\parallel$ [nN]')
            ax6.set(xlabel='contact ($F_N$) [%]', ylabel='mean $F_\parallel$ [nN]')
            

            for j in range(len(F_N)):                
                color = get_color_value(F_N[j], np.min(F_N), np.max(F_N))
                
                ax3.plot(stretch_pct, Ff[:, j, group, 0], color = color, linewidth = linewidth, markersize = markersize, label = f'F_N = {F_N[j]:g}')    
                ax4.plot(stretch_pct, Ff[:, j, group, 1], color = color, linewidth = linewidth, markersize = markersize, label = f'F_N = {F_N[j]:g}')
              
                sortidx = np.argsort(contact_mean[:,j,1])
                ax7.plot(contact_mean[sortidx,j,1], Ff[sortidx, j, group, 0], color = color, linewidth = linewidth, markersize = markersize, label = f'F_N = {F_N[j]:g}')                
                ax8.plot(contact_mean[sortidx,j,1], Ff[sortidx, j, group, 1], color = color, linewidth = linewidth, markersize = markersize, label = f'F_N = {F_N[j]:g}')
                

            norm = matplotlib.colors.BoundaryNorm(F_N, cmap.N)
            cax = make_axes_locatable(ax4).append_axes("right", "5%")
            cax.grid(False)
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='$F_N$ [nN]')
            
            cax = make_axes_locatable(ax8).append_axes("right", "5%")
            cax.grid(False)
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='$F_N$ [nN]')
            
            ax3.set(xlabel='stretch [%]', ylabel='max $F_\parallel$ [nN]')
            ax4.set(xlabel='stretch [%]', ylabel='mean $F_\parallel$ [nN]')
            ax7.set(xlabel='contact(stretch) [%]', ylabel='max $F_\parallel$ [nN]')
            ax8.set(xlabel='contact(stretch) [%]', ylabel='mean $F_\parallel$ [nN]')

            plt.tight_layout()     
            obj_list.append(interactive_plotter(fig))
                

        # --- Contact --- #
        fig = plt.figure(num = unique_fignum())
        if show_filename: fig.suptitle(folder)
        grid = (2,2)
        ax11 = plt.subplot2grid(grid, (0, 0), colspan=1) # (stretch, full sheet contact)
        ax22 = plt.subplot2grid(grid, (0, 1), colspan=1) # (stretch, inner sheet contact)
        ax33 = plt.subplot2grid(grid, (1, 0), colspan=1) # (F_N, full sheet contact)
        ax44 = plt.subplot2grid(grid, (1, 1), colspan=1) # (F_N, inner sheet contact)


        ymin = np.min(contact_mean)
        for j in range(len(F_N)):                
                color = get_color_value(F_N[j], np.min(F_N), np.max(F_N))
                ax11.plot(stretch_pct, contact_mean[:,j,0], color = color, linewidth = linewidth, markersize = markersize, label = f'F_N = {F_N[j]:g}')    
                ax22.plot(stretch_pct, contact_mean[:,j,1], color = color, linewidth = linewidth, markersize = markersize, label = f'F_N = {F_N[j]:g}')
            
        ylim = (np.min(contact_mean[~np.isnan(contact_mean)]), np.max(contact_mean[~np.isnan(contact_mean)]))

        ax11.set(xlabel='stretch [%]', ylabel='contact (full sheet) [%]')
        ax11.set_ylim(ylim)

        ax22.set_ylim(ylim)
        ax22.set(xlabel='stretch [%]', ylabel='contact (inner sheet) [%]')       

        norm = matplotlib.colors.BoundaryNorm(F_N, cmap.N)
        cax = make_axes_locatable(ax22).append_axes("right", "5%")
        cax.grid(False)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='$F_N$ [nN]')

                    
        for i in range(len(stretch_pct)):
                color = get_color_value(stretch_pct[i], np.min(stretch_pct), np.max(stretch_pct), cmap=cmap)
                ax33.plot(F_N, contact_mean[i, :, 0], color = color, linewidth = linewidth, label = f'stretch = {stretch_pct[i]:g}')
                ax44.plot(F_N, contact_mean[i, :, 1], color = color, linewidth = linewidth, label = f'stretch = {stretch_pct[i]:g}')
                

        ylim = (np.min(contact_mean[~np.isnan(contact_mean)]), np.max(contact_mean[~np.isnan(contact_mean)]))

        ax33.set(xlabel='$F_N$ [nN]', ylabel='contact (full sheet) [%]')
        ax33.set_ylim(ylim)

        ax44.set_ylim(ylim)
        ax44.set(xlabel='$F_N$ [nN]', ylabel='contact (inner sheet) [%]')       

        norm = matplotlib.colors.BoundaryNorm(stretch_pct, cmap.N)
        cax = make_axes_locatable(ax44).append_axes("right", "5%")
        cax.grid(False)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='Stretch [%]')
            
        plt.tight_layout()       
        obj_list.append(interactive_plotter(fig))
        
        
        
        # --- Rupture --- #
        fig = plt.figure(num = unique_fignum())
        if show_filename: fig.suptitle(folder)
        
        rup_true = np.argwhere(rup == 1)
        rup_false = np.argwhere(rup == 0)
        
        
       
        plt.scatter(rup_stretch_pct[rup_false[:,0]], rup_F_N[rup_false[:,1]], marker = 'o', label = "Intact")
        plt.scatter(rup_stretch_pct[rup_true[:,0]], rup_F_N[rup_true[:,1]], marker = 'x', label = "Ruptured")
        plt.xlabel('Stretch [%]')
        plt.ylabel('$F_N$ [nN]')
        plt.legend()
        

        
    return obj_list





def stability_heatmap(folders, mean_pct = 0.5, std_pct = 0.01, eval_rupture = False):
    stretch_lim = [None, 0.23]
    FN_lim = [None, 220]
    for folder in folders:
        stretch_pct, F_N, Ff, Ff_std, rup, filenames, contact_mean, contact_std = read_multi_folder(folder, mean_pct, std_pct, eval_rupture, stretch_lim, FN_lim)
        
        print()
        plot_heatmap( Ff_std[:, :, 0],
                     ['Stretch [%]', stretch_pct], 
                     ['$F_N$ [nN]', F_N])
        

        plt.show()
        
        

if __name__ == "__main__":
    # folders = ['../Data/multi_fast']
    # folders = ['../Data/BIG_MULTI_Xdrag']
    # folders = ['../Data/BIG_MULTI_Ydrag']
    # folders = ['../Data/BIG_MULTI_nocut']
    # stability_heatmap(folders)
    
    # folders = ['../Data/CONFIGS/cut_n',
    #            '../Data/CONFIGS/cut_nocut/conf_1',
    #            '../Data/CONFIGS/cut_nocut/conf_2',
    #            '../Data/CONFIGS/cut_nocut/conf_4',
    #            ]
   
    # folders.pop(0)
    
    obj = plot_multi([ '../Data/CONFIGS/sizes/conf_5'])
    # stability_heatmap(folders)
    plt.show()
    