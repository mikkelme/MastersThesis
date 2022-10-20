from analysis_utils import *



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


    
def plot_info(filenames):
    # interval = 10
    # window_length = 20
    # polyorder = 5
    window_length = 50
    polyorder = 5


    for i, filename in enumerate(filenames):            
            # --- Get data --- #
            # Read from file
            timestep, f_move_force1, f_move_force2, c_Ff_sheet1, c_Ff_sheet2, c_Ff_sheet3, c_Ff_PB1, c_Ff_PB2, c_Ff_PB3, c_sheet_COM1, c_sheet_COM2, c_sheet_COM3 = read_friction_file(filename)
            
            
            # Find a way to get pulling direction and dt
            drag_direction = np.array((0, 1))
            dt = 0.001
            
            # perp, parallel, z
            
            time = timestep * dt # [ps]
            # Organize in columns: parallel to drag, perpendicular to drag, z-axis
            move_force = np.vstack((decompose_wrt_drag_dir(f_move_force1, f_move_force2, drag_direction), np.zeros(len(f_move_force1)))).T
            Ff_sheet = np.vstack((decompose_wrt_drag_dir(c_Ff_sheet1, c_Ff_sheet2, drag_direction), c_Ff_sheet3)).T
            Ff_PB = np.vstack((decompose_wrt_drag_dir(c_Ff_PB1, c_Ff_PB2, drag_direction), c_Ff_PB3)).T
            COM_sheet = np.vstack((decompose_wrt_drag_dir(c_sheet_COM1, c_sheet_COM2, drag_direction), c_sheet_COM3)).T
            COM_sheet -= COM_sheet[0,:] # origo as reference point
         
            # # Smoothen or average
            Ff_sheet[:,0], Ff_sheet[:,1], Ff_sheet[:,2], Ff_PB[:,0], Ff_PB[:,1], Ff_PB[:,2], move_force[:,0], move_force[:,1] = savgol_filter(window_length, polyorder, Ff_sheet[:,0], Ff_sheet[:,1], Ff_sheet[:,2], Ff_PB[:,0], Ff_PB[:,1], Ff_PB[:,2], move_force[:,0], move_force[:,1])
            
            # Fxy_norm = np.sqrt(c_Ff1**2 + c_Ff2**2)
            # move_force_norm = np.sqrt(move_force1**2 + move_force2**2)
            
            Ff_full_sheet = Ff_sheet + Ff_PB

            
            
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

            # -- Ff_para -- #
            # Move force and full sheet
            ax1.plot(time, move_force[:,0], label = "move", color = color_cycle(0))
            ax1.plot(time, Ff_full_sheet[:,0], label = "group/group full_sheet", color = color_cycle(1))
            ax1.set(xlabel='Time $[ps]$', ylabel='$F_\parallel$ $[eV/Å]$')

            # Force decomposition: Full_sheet = sheet + PB
            ax2.plot(time, Ff_full_sheet[:,0], color = color_cycle(1))
            ax2.plot(time, Ff_sheet[:,0], label = "group/group sheet", color = color_cycle(2))
            ax2.plot(time, Ff_PB[:,0], label = "group/group PB", color = color_cycle(3))
            ax2.set(xlabel='Time $[ps]$', ylabel='$F_\parallel$ $[eV/Å]$')
            
            
            
            # -- Ff_perp -- #
            # Move force and full sheet
            ax3.plot(time, move_force[:,1], color = color_cycle(0))
            ax3.plot(time, Ff_full_sheet[:,1], color = color_cycle(1))
            ax3.set(xlabel='Time $[ps]$', ylabel='$F_\perp$ $[eV/Å]$')

            # Force decomposed: Full_sheet = sheet + PB
            ax4.plot(time, Ff_full_sheet[:,1], color = color_cycle(1))
            ax4.plot(time, Ff_sheet[:,1], color = color_cycle(2))
            ax4.plot(time, Ff_PB[:,1], color = color_cycle(3))
            ax4.set(xlabel='Time $[ps]$', ylabel='$F_\perp$ $[eV/Å]$')
            
            
            # -- Normal force -- #
            ax5.plot(time, -Ff_full_sheet[:,2], color = color_cycle(1))
            ax5.plot(time, -Ff_sheet[:,2], color = color_cycle(2))
            ax5.plot(time, -Ff_PB[:,2], color = color_cycle(3))
            ax5.set(xlabel='Time $[ps]$', ylabel='$F_N$ $[eV/Å]$')
            
        
            # -- COM -- # 
            # Decomposed: COM = parallel + perpendicular
            ax6.plot(time, COM_sheet[:,0], label = "$COM_\parallel$", color = color_cycle(4))
            ax6.plot(time, COM_sheet[:,1], label = "$COM_\perp$", color = color_cycle(5))
            ax6.set(xlabel='Time $[ps]$', ylabel='$\Delta COM$ $[Å]$')

            # Top view 
            plot_xy_time(fig, ax7, COM_sheet[:,0], COM_sheet[:,1], time)
            ax7.axis('equal')
            ax7.set(xlabel='$\Delta COM_\parallel$ $[Å]$', ylabel='$\Delta COM_\perp$ $[Å]$')
            # Put label on colorbar!
            
            fig.legend(loc = 'lower center', ncol=3, fancybox = True, shadow = True)
            # fig.legend(bbox_to_anchor=(0.5, 1.35), loc="upper center", bbox_transform=fig.transFigure, ncol=3, fancybox = True, shadow = True)
            
            fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
            # fig.savefig('image_output.png', bbox_inches='tight')
            
            
            

     
            # --- Calculate properties ---  #
            # 1521 atoms in group full_sheet
            # 360 atoms in group PB_tot

            # FN = np.mean(c_Ff3)
            # mu_avg = np.mean(Fxy_norm)/abs(FN)
            # print(f"mu_avg = {mu_avg:.2e}, mu_max = {mu_max:.2e}, (file = {filename}")
            
            # Friction coefficient (max)
            FN_full_sheet = np.mean(Ff_full_sheet[:,2])
            FN_sheet = np.mean(Ff_sheet[:,2])
            FN_PB = np.mean(Ff_PB[:,2])
            
            mu = np.array((3, 2))
        
            mu_max_full_sheet = Ff_full_sheet[:,0].max()/abs(FN_full_sheet)
            mu_avg_full_sheet = np.mean(Ff_full_sheet[:,0])/abs(FN_full_sheet)
            
            mu_max_sheet = Ff_sheet[:,0].max()/abs(FN_sheet)
            mu_avg_sheet = np.mean(Ff_sheet[:,0])/abs(FN_sheet)
            
            mu_max_PB = Ff_PB[:,0].max()/abs(FN_PB)
            mu_avg_PB = np.mean(Ff_PB[:,0])/abs(FN_PB)
            
            mu = np.array([[mu_max_full_sheet, mu_avg_full_sheet],
                          [mu_max_sheet, mu_avg_sheet],
                          [mu_max_PB, mu_avg_PB]])
     
            if True: # Terminal table 
                data = pandas.DataFrame(mu, np.array(['full_sheet', 'sheet', 'PB']), np.array(['max', 'avg']))
                print(f"Filename: {filename}")
                print(data)
                print()
            else:  # Spreadsheet format
                print(f"{filename}; {mu_max_full_sheet}; {mu_max_sheet}; {mu_max_PB}; {mu_avg_full_sheet}; {mu_avg_sheet}; {mu_avg_PB}")
            





if __name__ == "__main__":

    filenames = []


    # filenames += get_files_in_folder('../Data/NewGreat4_K0/', ext = ".txt")
    filenames += get_files_in_folder('../Data/NewGreat4_dt05fs/', ext = ".txt")
    
    # filenames = ['../Data/great4/friction_force_cut_20stretch.txt', '../Data/great4_1ms/friction_force_cut_20stretch.txt', '../Data/great4_dt05fs/friction_force_cut_20stretch.txt']
    
    
    plot_info(filenames)
    plt.show()






  