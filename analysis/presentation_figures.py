from analysis_utils import *
from multi_data_reader import read_multi_folder
from mpl_toolkits.axes_grid1 import make_axes_locatable


def friction_plot(filename):
    data = analyse_friction_file(filename)    
    time = data['time']
    COM = data['COM_sheet'][:,0]
    Ff = data['Ff_full_sheet'][:, 0]
    
    window_len_pct = 0.5
    
    # TRIM
    map = np.argwhere(time < 1000).ravel()
    time = time[map]
    Ff = Ff[map]    
    COM = COM[map]    
    
    
    plt.figure(num = unique_fignum())
    ax = plt.gca()
    
    
    plt.plot(time, Ff, color = color_cycle(0))
    plt.plot(time, cum_mean(Ff), color = color_cycle(1), label = "cumulative mean")
    plt.plot(time, running_mean(Ff, window_len = int(window_len_pct*len(Ff)))[0], color = color_cycle(2), label = f"running mean ({window_len_pct*100:g}%)")
    plt.plot(time, cum_max(Ff), label = "cumulative max", color = color_cycle(3))
    
    plt.legend()

    plt.xlabel("Time [ps]")
    plt.ylabel("$F_{f,\parallel}$ [nN]")
    add_xaxis(ax, time, COM, xlabel='COM$\parallel$ [Å]', decimals = 1) 
    
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    # plt.savefig("../Presentation/figures/drag2.pdf", bbox_inches="tight")
    
def contact_plot(filename):
    data = analyse_friction_file(filename)    
    time = data['time']
    COM = data['COM_sheet'][:,0]
    contact = data['contact'][0]
    
    window_len_pct = 0.5
    
    # TRIM
    # map = np.argwhere(time < 1000).ravel()
    # time = time[map]
    # contact = contact[map]    
    # COM = COM[map]    
    
    plt.figure(num = unique_fignum())
    ax = plt.gca()
    
    
    plt.plot(time, contact, color = color_cycle(0))
    plt.plot(time, cum_mean(contact), color = color_cycle(1), label = "cumulative mean")
    plt.plot(time, running_mean(contact, window_len = int(window_len_pct*len(contact)))[0], color = color_cycle(2), label = f"running mean ({window_len_pct*100:g}%)")
    
    plt.legend()

    plt.xlabel("Time [ps]")
    plt.ylabel("Bond count [%]")
    add_xaxis(ax, time, COM, xlabel='COM$\parallel$ [Å]', decimals = 1) 
    
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.savefig("../Presentation/figures/contact2.pdf", bbox_inches="tight")
    
def multi_plot_compare(folder1, folder2):
    folders = [folder1, folder2]
    grid = (1,2)
    
    fig1 = plt.figure(figsize = (10,5), num = unique_fignum())
    ax11 = plt.subplot2grid(grid, (0, 0), colspan=1)
    ax12 = plt.subplot2grid(grid, (0, 1), colspan=1)
    
    fig2 = plt.figure(figsize = (10,5), num = unique_fignum())
    ax21 = plt.subplot2grid(grid, (0, 0), colspan=1)
    ax22 = plt.subplot2grid(grid, (0, 1), colspan=1)
    
    fig3 = plt.figure(figsize = (10,5), num = unique_fignum())
    ax31 = plt.subplot2grid(grid, (0, 0), colspan=1)
    ax32 = plt.subplot2grid(grid, (0, 1), colspan=1)
    
    fig4 = plt.figure(figsize = (10,5), num = unique_fignum())
    ax41 = plt.subplot2grid(grid, (0, 0), colspan=1)
    ax42 = plt.subplot2grid(grid, (0, 1), colspan=1)
    
    fig5 = plt.figure(figsize = (10,5), num = unique_fignum())
    ax51 = plt.subplot2grid(grid, (0, 0), colspan=1)
    ax52 = plt.subplot2grid(grid, (0, 1), colspan=1)
    
    fig6 = plt.figure(figsize = (10,5), num = unique_fignum())
    ax61 = plt.subplot2grid(grid, (0, 0), colspan=1)
    ax62 = plt.subplot2grid(grid, (0, 1), colspan=1)
    
    fig7 = plt.figure(figsize = (10,5), num = unique_fignum())
    ax71 = plt.subplot2grid(grid, (0, 0), colspan=1)
    ax72 = plt.subplot2grid(grid, (0, 1), colspan=1)
    
    cmap = matplotlib.cm.viridis
    
    name = [folder.split('/')[-2] for folder in folders]
    figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7]
    ax = np.array([[ax11, ax21, ax31, ax41, ax51, ax61, ax71], [ax12, ax22,  ax32, ax42, ax52, ax62, ax72]]).T
    for f, folder in enumerate(folders):
        (stretch_pct, F_N, Ff, Ff_std, contact_mean, contact_std), (rup_stretch_pct, rup_F_N, rup, filenames) = read_multi_folder(folder, mean_pct = 0.5, std_pct = 0.01)
        # stretch_pct, F_N, Ff, Ff_std, rup, filenames, contact = read_multi_folder(folder, eval_rupture = False, stretch_lim = [None, 0.22])
        group = 0
        
        
        for i in range(len(stretch_pct)):
            color = get_color_value(stretch_pct[i], np.min(stretch_pct), np.max(stretch_pct), cmap=cmap)
            
            ax[0, f].plot(F_N, Ff[i, :, group, 1], marker = 'o', markersize = 3, color = color, label = f'stretch = {stretch_pct[i]:g}')
            ax[0, f].set(xlabel='$F_N$ [nN]', ylabel='mean $F_\parallel$ [nN]')
            
            ax[4, f].plot(F_N, contact_mean[i, :, 0], marker = 'o', markersize = 3,  color = color, label = f'stretch = {stretch_pct[i]:g}')
            ax[4, f].set(xlabel='$F_N$ [nN]', ylabel='Bond count [%]')
            
            ax[5, f].plot(contact_mean[i, :, 0], Ff[i, :, group, 1], marker = 'o', markersize = 3,  color = color, label = f'stretch = {stretch_pct[i]:g}')
            ax[5, f].set(xlabel='Bond count [%] (variable $F_N$)', ylabel='mean $F_\parallel$ [nN]')

    
    
        for j in range(len(F_N)):
            color = get_color_value(F_N[j], np.min(F_N), np.max(F_N))
        
            ax[1, f].plot(stretch_pct, Ff[:, j, group, 0], marker = 'o', markersize = 3,  color = color, label = f'F_N = {F_N[j]:g}')
            ax[1, f].set(xlabel='stretch [%]', ylabel='max $F_\parallel$ [nN]')
            
            ax[2, f].plot(stretch_pct, Ff[:, j, group, 1], marker = 'o', markersize = 3,  color = color, label = f'F_N = {F_N[j]:g}')
            ax[2, f].set(xlabel='stretch [%]', ylabel='mean $F_\parallel$ [nN]')
            
            ax[3, f].plot(stretch_pct, contact_mean[:, j, 0], marker = 'o', markersize = 3,  color = color, label = f'F_N = {F_N[j]:g}')
            ax[3, f].set(xlabel='stretch [%]', ylabel='Bond count [%]')
            
            ax[6, f].plot(contact_mean[:, j, 0], Ff[:, j, group, 1], marker = 'o', markersize = 3,  color = color, label = f'F_N = {F_N[j]:g}')
            ax[6, f].set(xlabel='Bond count [%] (variable stretch)', ylabel='mean $F_\parallel$ [nN]')
            
    
    
    for i, fig in enumerate(figs): # figures
        if i in [0, 4, 5]:
            norm = matplotlib.colors.BoundaryNorm(stretch_pct, cmap.N)
            label = 'stretch [%]'
        else:
            label = '$F_N$ [nN]'
            norm = matplotlib.colors.BoundaryNorm(F_N, cmap.N)
        cax = make_axes_locatable(ax[i, 1]).append_axes("right", "5%")
        cax.grid(False)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=label)
        
        ylim = ax[i, 0].get_ylim() + ax[i, 1].get_ylim()
        ylim = [np.min(ylim), np.max(ylim)]
        
        for j in range(2): 
            ax[i, j].set_ylim(ylim)
            ax[i, j].set_title(name[j])
            
        fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        # fig.savefig(f"../Presentation/figures/multi_lowFN{i}.pdf", bbox_inches="tight")

def multi_plot_max_mean(folder):
    grid = (1,2)
    
    fig = plt.figure(figsize = (10,5), num = unique_fignum())
    fig.suptitle("Cuts")
    ax1 = plt.subplot2grid(grid, (0, 0), colspan=1)
    ax2 = plt.subplot2grid(grid, (0, 1), colspan=1)
       
    cmap = matplotlib.cm.viridis
    
    
    stretch_pct, F_N, Ff, Ff_std, rup, filenames, contact = read_multi_folder(folder, eval_rupture = False, stretch_lim = [None, 0.22], FN_lim = [None, 100])
    group = 0
    
    for j in range(len(F_N)):
        color = get_color_value(F_N[j], np.min(F_N), np.max(F_N))
    
        ax1.plot(stretch_pct, Ff[:, j, group, 0], marker = 'o', markersize = 3,  color = color, label = f'F_N = {F_N[j]:g}')
        ax1.set(xlabel='stretch [%]', ylabel='max $F_\parallel$ [nN]')
        
        ax2.plot(stretch_pct, Ff[:, j, group, 1], marker = 'o', markersize = 3,  color = color, label = f'F_N = {F_N[j]:g}')
        ax2.set(xlabel='stretch [%]', ylabel='mean $F_\parallel$ [nN]')
        
        

    norm = matplotlib.colors.BoundaryNorm(F_N, cmap.N)
    cax = make_axes_locatable(ax2).append_axes("right", "5%")
    cax.grid(False)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='$F_N$ [nN]')

        

    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    fig.savefig(f"../Presentation/figures/max_mean_highFN.pdf", bbox_inches="tight")

     

def multi_plot_groups(folder):
    grid = (1,3)
    
    fig = plt.figure(figsize = (10,5), num = unique_fignum())
    fig.suptitle("Cuts")
    ax1 = plt.subplot2grid(grid, (0, 0), colspan=1)
    ax2 = plt.subplot2grid(grid, (0, 1), colspan=1)
    ax3 = plt.subplot2grid(grid, (0, 2), colspan=1)
       
    cmap = matplotlib.cm.viridis
    
    
    stretch_pct, F_N, Ff, Ff_std, rup, filenames, contact = read_multi_folder(folder, eval_rupture = False, stretch_lim = [None, 0.22])
    group = 0
    group_name = {0: 'Full sheet', 1: 'Sheet', 2: 'Pull blocks'}
    
    ax = [ax1, ax2, ax3]
    
    for group in range(3):
        for j in range(len(F_N)):
            color = get_color_value(F_N[j], np.min(F_N), np.max(F_N))
        
            ax[group].set_title(group_name[group])
            ax[group].plot(stretch_pct, Ff[:, j, group, 0], marker = 'o', markersize = 3,  color = color, label = f'F_N = {F_N[j]:g}')
            ax[group].set(xlabel='stretch [%]', ylabel='max $F_\parallel$ [nN]')
            
            
            
    norm = matplotlib.colors.BoundaryNorm(F_N, cmap.N)
    cax = make_axes_locatable(ax3).append_axes("right", "5%")
    cax.grid(False)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='$F_N$ [nN]')

        
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    fig.savefig(f"../Presentation/figures/max_group.pdf", bbox_inches="tight")

          
     
if __name__ == "__main__":
    
    # filename = '../Data/Multi/nocuts/ref1/stretch_15000_folder/job2/system_drag_Ff.txt'
    # filename = '../Data/Multi/nocuts/ref2/stretch_15000_folder/job5/system_drag_Ff.txt'
    # friction_plot(filename)
    # contact_plot(filename)
    
    multi_plot_compare('../Data/CONFIGS/nocut_sizes/conf_2', '../Data/CONFIGS/cut_sizes/conf_2')
    # multi_plot_max_mean('../Data/Multi/cuts/ref2')
    # multi_plot_groups('../Data/Multi/cuts/ref3')
    plt.show()