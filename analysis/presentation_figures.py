from analysis_utils import *
from mpl_toolkits.axes_grid1 import make_axes_locatable



def friction_plot(filename):
    data = analyse_friction_file(filename)    
    time = data['time']
    COM = data['COM_sheet'][:,0]
    Ff = data['Ff_full_sheet'][:, 0]
    
    # TRIM
    map = np.argwhere(time < 1000).ravel()
    time = time[map]
    Ff = Ff[map]    
    COM = COM[map]    
    
    
    plt.figure(num = get_fignum())
    ax = plt.gca()
    
    
    plt.plot(time, Ff)
    plt.plot(time, cum_mean(Ff), label = "cumulative mean")
    plt.plot(time, cum_max(Ff), label = "cumulative max")
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
    
    # TRIM
    # map = np.argwhere(time < 1000).ravel()
    # time = time[map]
    # contact = contact[map]    
    # COM = COM[map]    
    
    plt.figure(num = get_fignum())
    ax = plt.gca()
    
    
    plt.plot(time, contact)
    plt.plot(time, cum_mean(contact), label = "cumulative mean")
    plt.legend()

    plt.xlabel("Time [ps]")
    plt.ylabel("Bond count [%]")
    add_xaxis(ax, time, COM, xlabel='COM$\parallel$ [Å]', decimals = 1) 
    
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.savefig("../Presentation/figures/contact2.pdf", bbox_inches="tight")
    
def multi_plot(folder):
    stretch_pct, F_N, Ff, Ff_std, rup, filenames, contact = read_multi_folder(folder, eval_rupture = False, stretch_lim = [None, 0.22])
    
    
    fig = plt.figure(figsize = (10,5), num = get_fignum())
    grid = (1,2)
    ax1 = plt.subplot2grid(grid, (0, 0), colspan=1)
    ax2 = plt.subplot2grid(grid, (0, 1), colspan=1)
    cmap = matplotlib.cm.viridis
    
    
    group = 0
    for j in range(len(F_N)):
        color = get_color_value(F_N[j], np.min(F_N), np.max(F_N))
        
        ax1.plot(stretch_pct, Ff[:, j, group, 0], marker = 'o', markersize = 3,  color = color, label = f'F_N = {F_N[j]:g}')
        ax2.plot(stretch_pct, Ff[:, j, group, 1], marker = 'o', markersize = 3,  color = color, label = f'F_N = {F_N[j]:g}')
    
         
    norm = matplotlib.colors.BoundaryNorm(F_N, cmap.N)
    cax = make_axes_locatable(ax2).append_axes("right", "5%")
    cax.grid(False)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='$F_N$ [nN]')
       
    ax1.set_ylim([0, 8.5])
    ax2.set_ylim([0, 0.9])
    ax1.set(xlabel='stretch [%]', ylabel='max $F_\parallel$ [nN]')
    ax2.set(xlabel='stretch [%]', ylabel='mean $F_\parallel$ [nN]')
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.savefig("../Presentation/figures/multi_cuts.pdf", bbox_inches="tight")

     
if __name__ == "__main__":
    
    # filename = '../Data/Multi/nocuts/ref1/stretch_15000_folder/job2/system_drag_Ff.txt'
    # filename = '../Data/Multi/nocuts/ref2/stretch_15000_folder/job5/system_drag_Ff.txt'
    # friction_plot(filename)
    # contact_plot(filename)
    
    multi_plot('../Data/Multi/cuts/ref2')
    plt.show()