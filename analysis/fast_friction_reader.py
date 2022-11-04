from analysis_utils import *

from matplotlib.widgets import Button, TextBox, Slider

def plot_xy_time(fig, ax, x,y,time):
    """ Plot 2D x,y-plot with colorbar for time devolopment """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(time.min(), time.max())
    lc = LineCollection(segments, cmap='gist_rainbow', norm=norm)

    # Set the values used for colormapping
    lc.set_array(time)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    cbar = fig.colorbar(line, ax=ax)
   
    cbar.set_label('Time $[ps]$', rotation=270, labelpad = 20)
   
 
    # Set limits
    xsp = np.abs(x.max() - x.min()) * 0.1
    ysp = np.abs(y.max() - y.min()) * 0.1 
    if xsp != 0: ax.set_xlim(x.min() - xsp, x.max() + xsp)
    if ysp != 0: ax.set_ylim(y.min() - ysp, y.max() + ysp)


    
class interactive_plotter:
    """ Gets glitchy with multiple big figures open """
    def __init__(self, fig):
        self.cid_pick = fig.canvas.mpl_connect('button_press_event', self.pick_figure)
        self.fig = fig
        self.zoom = False
        self.ax_list = fig.axes
    
    def pick_figure(self, event):
        # print("clicked")
        if not self.zoom:
            if event.inaxes is not None:
                
                # col = event.inaxes.collections  
                # if len(col) > 0: 
                #     col[-1].set(visible = False)
                #     self.cb = col[-1].colorbar 

                self.old_axes, self.old_pos = event.inaxes, event.inaxes.get_position()
                pad = 0.1
                event.inaxes.set_position([pad, pad, 1-2*pad, 1-2*pad]) 
                self.toggle_axes(self.ax_list)
                self.toggle_axes([event.inaxes], visible = True)
                self.zoom = True
                
        else:
            if event.inaxes is None:
                self.toggle_axes(self.ax_list, visible = True)
                self.old_axes.set_position(self.old_pos)
                self.zoom = False
        self.fig.canvas.draw_idle()
        
    
    def toggle_axes(self, ax_list, visible = False):
        for ax in ax_list:
            ax.set_visible(visible)
            
           
def plot_info(filenames):
    window_length = 50
    polyorder = 5

    obj_list = []
    for i, filename in enumerate(filenames):            
            data = analyse_friction_file(filename)
            
            
            # fig = plt.figure(num = i+len(filenames))
            # plt.plot(data['time'], data['contact'][0], label = "full")
            # plt.plot(data['time'], data['contact'][1], label = "inner")
            # plt.ylabel('contact bonds [%]')
            # plt.xlabel('Time $[ps]$')
            # plt.legend()
            
            # fig = plt.figure(num = i+len(filenames)+1)
            # plt.plot(data['time'], data['Ff_full_sheet'][:,0]/data['contact'][0], label = "full")
            # plt.plot(data['time'],  data['Ff_sheet'][:,0]/data['contact'][1], label = "inner")
            # plt.xlabel('Time $[ps]$')
            # plt.ylabel('$F_\parallel$ / contact bonds $[eV/Å]$')
            # plt.legend()
            
            # --- Plotting --- #
            fig = plt.figure(num = i)
            fig.suptitle(filename)
            grid = (4,2)
            ax1 = plt.subplot2grid(grid, (0, 0), colspan=1)
            ax2 = plt.subplot2grid(grid, (0, 1), colspan=1)
            ax3 = plt.subplot2grid(grid, (1, 0), colspan=1)
            ax4 = plt.subplot2grid(grid, (1, 1), colspan=1)
            ax5 = plt.subplot2grid(grid, (2, 0), colspan=1)
            ax6 = plt.subplot2grid(grid, (2, 1), colspan=1)
            ax7 = plt.subplot2grid(grid, (3, 0), colspan=2)
            obj_list.append(interactive_plotter(fig))
            

            # -- Ff_para -- #
            # Move force and full sheet
            ax1.plot(data['time'], data['move_force'][:,0], label = "move", color = color_cycle(0))
            ax1.plot(data['time'], data['Ff_full_sheet'][:,0], label = "group/group full_sheet", color = color_cycle(1))
            ax1.set(xlabel='Time $[ps]$', ylabel='$F_\parallel$ $[eV/Å]$')

            # Force decomposition: Full_sheet = sheet + PB
            ax2.plot(data['time'], data['Ff_full_sheet'][:,0], color = color_cycle(1))
            ax2.plot(data['time'], data['Ff_sheet'][:,0], label = "group/group sheet", color = color_cycle(2))
            ax2.plot(data['time'], data['Ff_PB'][:,0], label = "group/group PB", color = color_cycle(3))
            ax2.set(xlabel='Time $[ps]$', ylabel='$F_\parallel$ $[eV/Å]$')
            
            
            
            # -- Ff_perp -- #
            # Move force and full sheet
            ax3.plot(data['time'], data['move_force'][:,1], color = color_cycle(0))
            ax3.plot(data['time'], data['Ff_full_sheet'][:,1], color = color_cycle(1))
            ax3.set(xlabel='Time $[ps]$', ylabel='$F_\perp$ $[eV/Å]$')

            # Force decomposed: Full_sheet = sheet + PB
            ax4.plot(data['time'], data['Ff_full_sheet'][:,1], color = color_cycle(1))
            ax4.plot(data['time'], data['Ff_sheet'][:,1], color = color_cycle(2))
            ax4.plot(data['time'], data['Ff_PB'][:,1], color = color_cycle(3))
            ax4.set(xlabel='Time $[ps]$', ylabel='$F_\perp$ $[eV/Å]$')
            
            
            # -- Normal force -- #
            ax5.plot(data['time'], -data['Ff_full_sheet'][:,2], color = color_cycle(1))
            ax5.plot(data['time'], -data['Ff_sheet'][:,2], color = color_cycle(2))
            ax5.plot(data['time'], -data['Ff_PB'][:,2], color = color_cycle(3))
            ax5.set(xlabel='Time $[ps]$', ylabel='$F_N$ $[eV/Å]$')
            
        
            # -- COM -- # 
            # Decomposed: COM = parallel + perpendicular
            ax6.plot(data['time'], data['COM_sheet'][:,0], label = "$COM_\parallel$", color = color_cycle(4))
            ax6.plot(data['time'], data['COM_sheet'][:,1], label = "$COM_\perp$", color = color_cycle(5))
            ax6.set(xlabel='Time $[ps]$', ylabel='$\Delta COM$ $[Å]$')

            # Top view 
            plot_xy_time(fig, ax7, data['COM_sheet'][:,0], data['COM_sheet'][:,1], data['time'])
            ax7.axis('equal')
            ax7.set(xlabel='$\Delta COM_\parallel$ $[Å]$', ylabel='$\Delta COM_\perp$ $[Å]$')
            
            fig.legend(loc = 'lower center', ncol=3, fancybox = True, shadow = True)
            # fig.legend(bbox_to_anchor=(0.5, 1.35), loc="upper center", bbox_transform=fig.transFigure, ncol=3, fancybox = True, shadow = True)
            fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
            # fig.savefig('image_output.png', bbox_inches='tight')
            
            
            
            #####
            
            mu = np.array((3, 2))
            
            mu_max_full_sheet = data['Ff'][0,0]/abs(data['FN'][0])
            mu_avg_full_sheet = data['Ff'][0,1]/abs(data['FN'][0])
            
            mu_max_sheet = data['Ff'][1,0]/abs(data['FN'][1])
            mu_avg_sheet = data['Ff'][1,1]/abs(data['FN'][1])
            
            mu_max_PB = data['Ff'][2,0]/abs(data['FN'][2])
            mu_avg_PB = data['Ff'][2,1]/abs(data['FN'][2])
            
            mu = np.array([[mu_max_full_sheet, mu_avg_full_sheet],
                          [mu_max_sheet, mu_avg_sheet],
                          [mu_max_PB, mu_avg_PB]])
     
     
     
            if True: # Terminal table 
                data = pandas.DataFrame(mu, np.array(['full_sheet', 'sheet', 'PB']), np.array(['max', 'avg']))
                print(f"Filename: {filename}")
                print(data)
                print()
            else:  # Spreadsheet format
                print(f"{filename} {mu_max_full_sheet} {mu_max_sheet} {mu_max_PB} {mu_avg_full_sheet} {mu_avg_sheet} {mu_avg_PB}")
            
            # plt.show()
       

if __name__ == "__main__":

    filenames = []


    # filenames += get_files_in_folder('../Data/NG4_newpot_long/', ext = "Ff.txt")
    filenames += get_files_in_folder('../Data/NG4_newpot_K0/', ext = "Ff.txt")
    
    
    # read_friction_file_dict('../friction_simulation/system_test_Ff.txt')
    # exit()
    
    
    filenames = ['../friction_simulation/system_test_Ff.txt', filenames[0], filenames[1]] 
    plot_info(filenames)
    plt.show()






  