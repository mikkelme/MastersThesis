import sys
sys.path.append('../') # parent folder: MastersThesis
from plot_set import *
import numpy as np
import matplotlib.pyplot as plt
from contact_area import plot_contact_area
from hysteresis import *
from utilities import *


def get_normal_buckling(sheet_dump, quartiles = [0.01, 0.05, 0.1, 0.25, 0.50]):
    """ 
    Measure buckling of sheet into plane normal direction (z). 
    Represented with min, max and quartile values of z_position.
    Does automatic sorting, meadian detection and exclusion of quartiles > 0.5.

    Return:
    timestep: data timesteps 
    Q: result values,  Q[:] = [min, ... 1 -quartiles ..., (median), ... quartiles ...]
    Q_var: result labels
    """
    # --- Get data --- #
    sheet_infile = open(sheet_dump, "r")
    timestep = []
    zpos = []
    print("# --- Processing normal buckling --- # ")
    print(f"Dump file = \"{sheet_dump}\"")




    while True: # Timestep loop
        try: 
            # --- Sheet positions --- #
            info = [sheet_infile.readline() for i in range(9)]
            if info[0] == '': break
            sheet_timestep = int(info[1].strip("\n"))
            # if sheet_timestep == 100000:  break
    

            sheet_num_atoms = int(info[3].strip("\n"))
            sheet_atom_pos = np.zeros((sheet_num_atoms, 3))
            print(f"\rTimestep = {sheet_timestep}", end = "")

            for i in range(sheet_num_atoms): # sheet atom loop
                line = sheet_infile.readline() # id type x y z [...]
                words = np.array(line.split(), dtype = float)
                sheet_atom_pos[i] = words[2:5] # <--- Be aware of different dumpt formats!
           
        except KeyboardInterrupt: break


        timestep.append(sheet_timestep)
        zpos.append(sheet_atom_pos[:,2])
    sheet_infile.close() # Done reading
    timestep = np.array(timestep)
    zpos = np.array(zpos)
    print() 
 
    # --- Prepare data preocessing --- #
    z0 = np.mean(zpos[0])
    zpos -= z0

    # Ensure valid quartiles order and values
    quartiles = np.sort(quartiles) # Sort
    quartiles = quartiles[quartiles <= 0.5] # Remove values higher than 0.5

    # Create matrix
    Q_len = 2*len(quartiles) + 2 # Quartiles + min, max
    if quartiles[-1] == 0.5: Q_len -= 1 # Avoid dublicates of median 
    Q = np.zeros((Q_len, zpos.shape[0]))

    # --- Calculate min, max and quartiles --- #
    # Max
    Q_var = ["Max"] 
    Q[0] = np.max(zpos, axis=-1)

    # Upper quartiles
    for i in range((Q_len-2)//2):
        Q_var.append("Q = " + str(1-quartiles[i]))
        Q[i+1] = np.quantile(zpos, 1-quartiles[i], axis = -1)

    # Median
    if (Q_len-2)%2: 
        # Q_var.append(str(quartiles[-1]))
        Q_var.append("Median")
        Q[i+2] = np.quantile(zpos, 1-quartiles[i+1], axis = -1)

    # Lower quartiles
    for i in reversed(range((Q_len-2)//2)):
        Q_var.append("Q = " + str(quartiles[i]))
        Q[-i-2] = np.quantile(zpos, quartiles[i], axis = -1)

    # Min
    Q[-1] = np.min(zpos, axis=-1)
    Q_var.append("Min")
    print()
    return timestep, Q_var, Q    
    



def normal_buckling(sheet_dump, stretch_file = None):
    # --- Get data --- #
    timestep, Q_var, Q = get_normal_buckling(sheet_dump)
    if stretch_file != None: timestamps = get_stretch_timestamps(stretch_file)

    # --- Plotting --- #
    # Relative to starting point

    ylabel = "Relative sheet z-position to starting position, ($z - z_0$)"
    for i in range(2):
        if i > 0:
            if len(Q)%2: # Relative to median
                med_idx = len(Q)//2
                Q -= Q[med_idx]
                ylabel = "Relative sheet z-position to smedian, ($z - z_{med}$)"
            else:
                continue

        plt.figure(num=i, dpi=80, facecolor='w', edgecolor='k')
        for i in range(Q.shape[0]):
            color = color_cycle(np.min((i, Q.shape[0]-i-1)))
            plt.plot(timestep, Q[i], color = color, label = Q_var[i])
        if stretch_file != None:
            ax = plt.gca()
            ylim = ax.get_ylim()
            plt.autoscale(False)
            for timestamp in timestamps:
                vline = plt.vlines(timestamp, ylim[0], ylim[1], linestyle = "--", color = "k")
            vline.set_label("Timestamps")
        plt.xlabel("Timestep", fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.legend(loc ='center left', bbox_to_anchor =(1, 0.5), fontsize = 13)
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)







if __name__ == "__main__":
    # sheet_dump = "../area_vs_stretch/sheet.data";
    # stretch_file = "../area_vs_stretch/stretch.txt";
    sheet_dump = "../Data/sheet_vacuum_bigfacet1/sheet_vacuum.data";
    stretch_file = "../Data/sheet_vacuum_bigfacet1/stretch.txt";
    

    normal_buckling(sheet_dump, stretch_file = stretch_file)
    plt.show()