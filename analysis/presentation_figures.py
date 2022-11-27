from analysis_utils import *


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
    
    
    fignum = 0
    if fignum in plt.get_fignums():
        fignum = plt.get_fignums()[-1] + 1
    plt.figure(num = fignum)
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
    
    fignum = 0
    if fignum in plt.get_fignums():
        fignum = plt.get_fignums()[-1] + 1
    plt.figure(num = fignum)
    ax = plt.gca()
    
    
    plt.plot(time, contact)
    plt.plot(time, cum_mean(contact), label = "cumulative mean")
    plt.legend()

    plt.xlabel("Time [ps]")
    plt.ylabel("Bond count [%]")
    add_xaxis(ax, time, COM, xlabel='COM$\parallel$ [Å]', decimals = 1) 
    
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.savefig("../Presentation/figures/contact2.pdf", bbox_inches="tight")
    
       
    
if __name__ == "__main__":
    
    # filename = '../Data/Multi/nocuts/ref1/stretch_15000_folder/job2/system_drag_Ff.txt'
    filename = '../Data/Multi/nocuts/ref2/stretch_15000_folder/job5/system_drag_Ff.txt'
    # friction_plot(filename)
    contact_plot(filename)
    plt.show()