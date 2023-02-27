from analysis_utils import *
from rupture_detect import *
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable

from read_stuff import *

def read_info_file_old(filename):
    stretch_pct, F_N = np.loadtxt(filename, unpack=True, delimiter = ',')
    return stretch_pct, F_N
    



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





def stability_heatmap(folders, mean_pct = 0.5, std_pct = 0.01, stretch_lim = [None, None],  FN_lim = [None, None]):
    stretch_lim = [None, 0.23]
    FN_lim = [None, 220]
    for folder in folders:
        (stretch_pct, F_N, Ff, Ff_std, contact_mean, contact_std), (rup_stretch_pct, rup_F_N, rup, filenames) = read_multi_folder(folder, mean_pct, std_pct, stretch_lim, FN_lim)
        # stretch_pct, F_N, Ff, Ff_std, rup, filenames, contact_mean, contact_std = read_multi_folder(folder, mean_pct, std_pct, eval_rupture, stretch_lim, FN_lim)
        
        
        print()
        ax = plot_heatmap( Ff_std[:, :, 0],
                     ['Stretch [%]', stretch_pct], 
                     ['$F_N$ [nN]', F_N])
        
        
        # Add markers for missing files (M) and ruptures (X)
        missing_map = np.argwhere(np.isnan(rup)).T + 0.5 
        rupture_map = np.argwhere(rup == 1).T + 0.5
        ax.scatter(rupture_map[0, :], rupture_map[1, :], marker="x", color="black", s=100)        
        ax.scatter(missing_map[0, :], missing_map[1, :], marker="$M$", color="black", s=100)        


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
    
    # obj = plot_multi([ '../Data/CONFIGS/cut_sizes/conf'])
    # stability_heatmap([ '../Data/CONFIGS/cut_sizes/conf_6'])
    # plt.show()
    
    
    data = read_multi_folder('../Data/CONFIGS/honeycomb/hon_21/')