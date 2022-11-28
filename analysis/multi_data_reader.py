from analysis_utils import *
from rupture_detect import *
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable

def read_info_file_old(filename):
    stretch_pct, F_N = np.loadtxt(filename, unpack=True, delimiter = ',')
    return stretch_pct, F_N
    


def read_multi_folder(folder, eval_rupture = False, stretch_lim = [None, None],  FN_lim = [None, None]):
    # Settings
    info_file = 'info_file.txt'
    friction_ext = 'Ff.txt'
    chist_ext = 'chist.txt'
    ruptol = 0 # 0.5
    
    avg_pct = 0.5 # last % to average over
    std_pct = 0
        
    data = []
    stretchfile = find_single_file(folder, ext = chist_ext)
    for a, stretch_dir in enumerate(get_dirs_in_path(folder)):
        alen = len(get_dirs_in_path(folder))
        for b, job_dir in enumerate(get_dirs_in_path(stretch_dir)):
            blen = len(get_dirs_in_path(stretch_dir))
            progress = a * blen + b
            total = alen * blen
            print(f"\r ({progress+1}/{total}) | {job_dir} ", end = " ")
            
            try:
                info_dict = read_info_file(os.path.join(job_dir,info_file))
                stretch_pct = info_dict['stretch_max_pct']
                F_N = metal_to_SI(info_dict['F_N'], 'F')*1e9
                
                
                if eval_rupture:
                    chist_file = find_single_file(job_dir, ext = chist_ext)
                    rupture_score = detect_rupture(chist_file, stretchfile)
                else: 
                    rupture_score = 0
                
                
                friction_file = find_single_file(job_dir, ext = friction_ext)     
                fricData = analyse_friction_file(friction_file, avg_pct, std_pct)
                data.append((stretch_pct, F_N, fricData['Ff'], fricData['Ff_std'], rupture_score, job_dir, fricData['contact_mean']))  
                
            except FileNotFoundError:
                print(f"<-- Missing file")
    print()
    
    
    data = np.array(data, dtype = 'object')
    stretch_pct, F_N, Ff, Ff_std, rup, filenames, contact = organize_data(data)
    
    # Trim to limits 
    if stretch_lim[0] == None: stretch_lim[0] = np.min(stretch_pct) - 1
    if stretch_lim[1] == None: stretch_lim[1] = np.max(stretch_pct) + 1
    if FN_lim[0] == None: FN_lim[0] = np.min(F_N) - 1
    if FN_lim[1] == None: FN_lim[1] = np.max(F_N) + 1
    
    stretch_args = np.argwhere(np.logical_and(stretch_lim[0] <= stretch_pct, stretch_pct <= stretch_lim[-1])).flatten()
    FN_args = np.argwhere(np.logical_and(FN_lim[0] <= F_N, F_N <= FN_lim[-1])).flatten()
    
    stretch_pct = stretch_pct[stretch_args]
    F_N = F_N[FN_args]
    Ff = Ff[stretch_args][:, FN_args]
    Ff_std = Ff_std[stretch_args][:, FN_args]
    rup = rup[stretch_args][:, FN_args] > ruptol
    filenames = filenames[stretch_args][:, FN_args]
    contact = contact[stretch_args][:, FN_args].astype('float')
    
    
    if eval_rupture:
        detections = [["stretch %", "F_N", "Filenames"]]
        for i in range(len(stretch_pct)):
            for j in range(len(F_N)):
                if rup[i,j] > ruptol:
                    detections.append([stretch_pct[i], F_N[j], filenames[i,j]])
        
        if len(detections) > 1:
            print("Rupture detected:")
            print(np.array(detections))
        else:
            print("No rupture detected")
    else: 
        print("eval_rupture = False")
    
    
    return stretch_pct, F_N, Ff, Ff_std, rup, filenames, contact


def plot_multi(folders, eval_rupture = False, stretch_lim = [None, None],  FN_lim = [None, None]):
    for folder in folders:
        stretch_pct, F_N, Ff, Ff_std, rup, filenames, contact = read_multi_folder(folder, eval_rupture, stretch_lim, FN_lim)
    
        group_name = {0: 'Full sheet', 1: 'Sheet', 2: 'PB'}
        linewidth = 1.5
        marker = 'o'
        markersize = 2.5

        rup_marker = 'x'
        rupmarkersize = markersize * 3
            

            
        obj_list = []
        for group in reversed(range(3)):
            # --- Plotting --- #
            fig = plt.figure(num = group)
            fig.suptitle(group_name[group])
            grid = (4,2)
            ax1 = plt.subplot2grid(grid, (0, 0), colspan=1)
            ax2 = plt.subplot2grid(grid, (0, 1), colspan=1)
            ax3 = plt.subplot2grid(grid, (1, 0), colspan=1)
            ax4 = plt.subplot2grid(grid, (1, 1), colspan=1)
            ax5 = plt.subplot2grid(grid, (2, 0), colspan=1)
            ax6 = plt.subplot2grid(grid, (2, 1), colspan=1)
            ax7 = plt.subplot2grid(grid, (3, 0), colspan=1)
            ax8 = plt.subplot2grid(grid, (3, 1), colspan=1)
            cmap = matplotlib.cm.viridis

            for i in range(len(stretch_pct)):
                color = get_color_value(stretch_pct[i], np.min(stretch_pct), np.max(stretch_pct), cmap=cmap)
                rup_true = np.argwhere(rup[i, :])
                rup_false = np.argwhere(~rup[i, :])
                
                ax1.plot(F_N, Ff[i, :, group, 0], color = color, linewidth = linewidth, label = f'stretch = {stretch_pct[i]:g}')
                ax1.plot(F_N[rup_true], Ff[i, rup_true, group, 0], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
                ax1.plot(F_N[rup_false], Ff[i, rup_false, group, 0], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
                
            
                ax2.plot(F_N, Ff[i, :, group, 1], color = color, linewidth = linewidth, label = f'stretch = {stretch_pct[i]:g}')
                ax2.plot(F_N[rup_true], Ff[i, rup_true, group, 1], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
                ax2.plot(F_N[rup_false], Ff[i, rup_false, group, 1], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
                
                
                sortidx = np.argsort(contact[i,:,1])
                ax5.plot(contact[i, sortidx,1], Ff[i, sortidx, group, 0], color = color, linewidth = linewidth, markersize = markersize, label = f'stretch = {stretch_pct[i]:g}')                
                ax5.plot(contact[i,sortidx[rup_true],1], Ff[i, sortidx[rup_true], group, 0], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
                ax5.plot(contact[i, sortidx[rup_false],1], Ff[i, sortidx[rup_false], group, 0], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
                
                ax6.plot(contact[i, sortidx,1], Ff[i, sortidx, group, 1], color = color, linewidth = linewidth, markersize = markersize, label = f'stretch = {stretch_pct[i]:g}')
                ax6.plot(contact[i, sortidx[rup_true],1], Ff[i, sortidx[rup_true], group, 1], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
                ax6.plot(contact[i, sortidx[rup_false],1], Ff[i, sortidx[rup_false], group, 1], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
                

                
                
            norm = matplotlib.colors.BoundaryNorm(stretch_pct, cmap.N)
            cax = make_axes_locatable(ax2).append_axes("right", "5%")
            cax.grid(False)
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='Stretch [%]')
            
            cax = make_axes_locatable(ax6).append_axes("right", "5%")
            cax.grid(False)
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='Stretch [%]')
            
            
            ax1.set(xlabel='$F_N$ [nN]', ylabel='max $F_\parallel$ [nN]')
            ax2.set(xlabel='$F_N$ [nN]', ylabel='mean $F_\parallel$ [nN]')
            ax5.set(xlabel='contact (sheet) [%]', ylabel='max $F_\parallel$ [nN]')
            ax6.set(xlabel='contact (sheet) [%]', ylabel='mean $F_\parallel$ [nN]')
            

            for j in range(len(F_N)):                
                color = get_color_value(F_N[j], np.min(F_N), np.max(F_N))
                rup_true = np.argwhere(rup[:, j])
                rup_false = np.argwhere(~rup[:, j])
                
                ax3.plot(stretch_pct, Ff[:, j, group, 0], color = color, linewidth = linewidth, markersize = markersize, label = f'F_N = {F_N[j]:g}')
                ax3.plot(stretch_pct[rup_true], Ff[rup_true, j, group, 0], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
                ax3.plot(stretch_pct[rup_false], Ff[rup_false, j, group, 0], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
                
                ax4.plot(stretch_pct, Ff[:, j, group, 1], color = color, linewidth = linewidth, markersize = markersize, label = f'F_N = {F_N[j]:g}')
                ax4.plot(stretch_pct[rup_true], Ff[rup_true, j, group, 1], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
                ax4.plot(stretch_pct[rup_false], Ff[rup_false, j, group, 1], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
                
                
                sortidx = np.argsort(contact[:,j,1])
                ax7.plot(contact[sortidx,j,1], Ff[sortidx, j, group, 0], color = color, linewidth = linewidth, markersize = markersize, label = f'F_N = {F_N[j]:g}')                
                ax7.plot(contact[sortidx[rup_true],j,1], Ff[sortidx[rup_true], j, group, 0], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
                ax7.plot(contact[sortidx[rup_false],j,1], Ff[sortidx[rup_false], j, group, 0], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
                
                ax8.plot(contact[sortidx,j,1], Ff[sortidx, j, group, 1], color = color, linewidth = linewidth, markersize = markersize, label = f'F_N = {F_N[j]:g}')
                ax8.plot(contact[sortidx[rup_true],j,1], Ff[sortidx[rup_true], j, group, 1], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
                ax8.plot(contact[sortidx[rup_false],j,1], Ff[sortidx[rup_false], j, group, 1], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
                

                
            norm = matplotlib.colors.BoundaryNorm(F_N, cmap.N)
            cax = make_axes_locatable(ax4).append_axes("right", "5%")
            cax.grid(False)
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='$F_N$ [nN]')
            
            cax = make_axes_locatable(ax8).append_axes("right", "5%")
            cax.grid(False)
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='$F_N$ [nN]')
            
            ax3.set(xlabel='stretch [%]', ylabel='max $F_\parallel$ [nN]')
            ax4.set(xlabel='stretch [%]', ylabel='mean $F_\parallel$ [nN]')
            ax7.set(xlabel='contact (sheet) [%]', ylabel='max $F_\parallel$ [nN]')
            ax8.set(xlabel='contact (sheet) [%]', ylabel='mean $F_\parallel$ [nN]')

            plt.tight_layout()     
            obj_list.append(interactive_plotter(fig))
                


        fig = plt.figure(num = 4)
        grid = (2,2)
        ax11 = plt.subplot2grid(grid, (0, 0), colspan=1)
        ax22 = plt.subplot2grid(grid, (0, 1), colspan=1)
        ax33 = plt.subplot2grid(grid, (1, 0), colspan=1)
        ax44 = plt.subplot2grid(grid, (1, 1), colspan=1)


        ymin = np.min(contact)
        for j in range(len(F_N)):                
                color = get_color_value(F_N[j], np.min(F_N), np.max(F_N))
                rup_true = np.argwhere(rup[:, j])
                rup_false = np.argwhere(~rup[:, j])
                
                
                ### XXX: GET contact max as well???
                ax11.plot(stretch_pct, contact[:,j,0], color = color, linewidth = linewidth, markersize = markersize, label = f'F_N = {F_N[j]:g}')
                ax11.plot(stretch_pct[rup_true], contact[rup_true, j, 0], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
                ax11.plot(stretch_pct[rup_false], contact[rup_false, j, 0], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
                
                ax22.plot(stretch_pct, contact[:,j,1], color = color, linewidth = linewidth, markersize = markersize, label = f'F_N = {F_N[j]:g}')
                ax22.plot(stretch_pct[rup_true], contact[rup_true, j, 1], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
                ax22.plot(stretch_pct[rup_false], contact[rup_false, j, 1], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
            
        ylim = (np.min(contact[~np.isnan(contact)]), np.max(contact[~np.isnan(contact)]))

        ax11.set(xlabel='stretch [%]', ylabel='contact (full sheet) [%]')
        ax11.set_ylim(ylim)

        ax22.set_ylim(ylim)
        ax22.set(xlabel='stretch [%]', ylabel='contact (sheet) [%]')       

        norm = matplotlib.colors.BoundaryNorm(F_N, cmap.N)
        cax = make_axes_locatable(ax22).append_axes("right", "5%")
        cax.grid(False)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='$F_N$ [nN]')

                    
        for i in range(len(stretch_pct)):
                color = get_color_value(stretch_pct[i], np.min(stretch_pct), np.max(stretch_pct), cmap=cmap)
                rup_true = np.argwhere(rup[i, :])
                rup_false = np.argwhere(~rup[i, :])
                
                ax33.plot(F_N, contact[i, :, 0], color = color, linewidth = linewidth, label = f'stretch = {stretch_pct[i]:g}')
                ax33.plot(F_N[rup_true], contact[i, rup_true, 0], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
                ax33.plot(F_N[rup_false], contact[i, rup_false, 0], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
                
                ax44.plot(F_N, contact[i, :, 1], color = color, linewidth = linewidth, label = f'stretch = {stretch_pct[i]:g}')
                ax44.plot(F_N[rup_true], contact[i, rup_true, 1], linestyle = 'None', marker = rup_marker, markersize = rupmarkersize, color=color)  
                ax44.plot(F_N[rup_false], contact[i, rup_false, 1], linestyle = 'None', marker = marker, markersize = markersize, color=color)  
                

        ylim = (np.min(contact[~np.isnan(contact)]), np.max(contact[~np.isnan(contact)]))

        ax33.set(xlabel='$F_N$ [nN]', ylabel='contact (full sheet) [%]')
        ax33.set_ylim(ylim)

        ax44.set_ylim(ylim)
        ax44.set(xlabel='$F_N$ [nN]', ylabel='contact (sheet) [%]')       

        norm = matplotlib.colors.BoundaryNorm(stretch_pct, cmap.N)
        cax = make_axes_locatable(ax44).append_axes("right", "5%")
        cax.grid(False)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='Stretch [%]')
            
        plt.tight_layout()       
        obj_list.append(interactive_plotter(fig))
    return obj_list
        # plt.show()





def stability_heatmap(folders, eval_rupture = False):
    stretch_lim = [None, 0.23]
    FN_lim = [None, 220]
    for folder in folders:
        stretch_pct, F_N, Ff, Ff_std, rup, filenames, contact = read_multi_folder(folder, eval_rupture, stretch_lim, FN_lim)
        
        
        plot_heatmap( Ff_std[:, :, 0, 1],
                     ['Stretch [%]', stretch_pct], 
                     ['$F_N$ [nN]', F_N])
        

        plt.show()

if __name__ == "__main__":
    # folders = ['../Data/multi_fast']
    # folders = ['../Data/BIG_MULTI_Xdrag']
    # folders = ['../Data/BIG_MULTI_Ydrag']
    # folders = ['../Data/BIG_MULTI_nocut']
    # stability_heatmap(folders)
    
    folders = ['../Data/Multi/cuts/ref2']
    obj = plot_multi(folders, False, stretch_lim = [None, 0.22])
    plt.show()
    