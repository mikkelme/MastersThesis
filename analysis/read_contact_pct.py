### Initially used scripts for
### for looking at contact vs time

from analysis_utils import *


def annotation_line( ax, xmin, xmax, y, text, ytext=0, linecolor='black', linewidth=1, fontsize=12 ):

    ax.annotate('', xy=(xmin, y), xytext=(xmax, y), xycoords='data', textcoords='data',
            arrowprops={'arrowstyle': '-', 'color':linecolor, 'linewidth':linewidth})
    ax.annotate('', xy=(xmin, y), xytext=(xmax, y), xycoords='data', textcoords='data',
            arrowprops={'arrowstyle': '<->', 'color':linecolor, 'linewidth':linewidth})

    xcenter = xmin + (xmax-xmin)/2
    if ytext==0:
        ytext = y + ( ax.get_ylim()[1] - ax.get_ylim()[0] ) / 20

    ax.annotate( text, xy=(xcenter,ytext), ha='center', va='center', fontsize=fontsize)

def read_contact_pct(filename):
    fignum = 0
    if fignum in plt.get_fignums():
        fignum = plt.get_fignums()[-1] + 1
    plt.figure(num = fignum)
    ax = plt.gca()
    
    timestep, sheet_bond_pct, full_sheet_bond_pct = np.loadtxt(filename, unpack=True)
    info = read_info_file('/'.join(filename.split('/')[:-1]) + '/info_file.txt' )
    dt = info['dt']
    time = timestep*dt
    
    relax = [0, info['relax_time']]
    stretch = [relax[-1], info['stretch_max_pct']/info['stretch_speed_pct'] + relax[-1]]
    pause1 = [stretch[-1], info['pause_time1'] + stretch[-1]]
    pause2 = [pause1[-1], info['pause_time2'] + pause1[-1]]
    drag = [pause2[-1], time[-1]]
    stages = [relax, stretch, pause1, pause2, drag]
    
    
    plt.plot(time,  full_sheet_bond_pct, label = "full_sheet bonds")
    plt.xlabel("Time [ps]")
    plt.ylabel("Rel. Contact")
    
    # xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    plt.vlines(0, ylim[0], ylim[1], "k", "--", alpha = 0.5, linewidth = 1)
    for stage in stages:
        plt.vlines(stage[1], ylim[0], ylim[1], "k", "--", alpha = 0.5, linewidth = 1)
    
    annotation_line(ax=ax, text='Relax', xmin=relax[0], xmax=relax[1], y=ylim[0], ytext=ylim[0]- 0.01, linewidth=1, linecolor='black', fontsize=12 )
    annotation_line(ax=ax, text='Stretch', xmin=stretch[0], xmax=stretch[1], y=ylim[0], ytext=ylim[0]- 0.01, linewidth=1, linecolor='black', fontsize=12 )
    annotation_line(ax=ax, text='Pause', xmin=pause1[0], xmax=pause1[1], y=ylim[0], ytext=ylim[0]- 0.01, linewidth=1, linecolor='black', fontsize=12 )
    annotation_line(ax=ax, text='$F_N$', xmin=pause2[0], xmax=pause2[1], y=ylim[0], ytext=ylim[0]- 0.01, linewidth=1, linecolor='black', fontsize=12 )
    annotation_line(ax=ax, text='Slide', xmin=drag[0], xmax=drag[1], y=ylim[0], ytext=ylim[0]- 0.01, linewidth=1, linecolor='black', fontsize=12 )
    
    ax.set_ylim([ylim[0]-0.02, ylim[1]])


    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.savefig("../presentation/figures/contact_pct.pdf", bbox_inches="tight")





if __name__ == "__main__":
    
    
    filename = '../Data/contact_stretch/bond_pct.txt'
    read_contact_pct(filename)
    plt.show()