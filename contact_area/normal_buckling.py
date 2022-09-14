import sys
sys.path.append('../') # parent folder: MastersThesis
from plot_set import *
import numpy as np
import matplotlib.pyplot as plt
from contact_area import plot_contact_area


def get_normal_buckling(sheet_dump, quartiles = [0.01, 0.05, 0.1, 0.25, 0.50]):
    """ Quartiles +- input, must be <= 0.5 = median """
    sheet_infile = open(sheet_dump, "r")

    timestep = []
    zpos = []
    while True: # Timestep loop
        # --- Sheet positions --- #
        info = [sheet_infile.readline() for i in range(9)]
        if info[0] == '': break
        sheet_timestep = int(info[1].strip("\n"))
        if sheet_timestep == 10000: break 
        sheet_num_atoms = int(info[3].strip("\n"))
        sheet_atom_pos = np.zeros((sheet_num_atoms, 3))
        print(f"\rtimestep = {sheet_timestep}", end = "")

        for i in range(sheet_num_atoms): # sheet atom loop
            line = sheet_infile.readline() # id type x y z [...]
            words = np.array(line.split(), dtype = float)
            sheet_atom_pos[i] = words[2:5] # <--- Be aware of different dumpt formats!
        
        timestep.append(sheet_timestep)
        zpos.append(sheet_atom_pos[:,2])
    sheet_infile.close() # Done reading
    print() 
    timestep = np.array(timestep)
    zpos = np.array(zpos)
 
    z0 = np.mean(zpos[0])
    zpos -= z0

    quartiles = np.sort(quartiles) # Sort
    quartiles = quartiles[quartiles <= 0.5] # Remove values higher than 0.5

    Q_len = 2*len(quartiles) + 2 # Quartiles + min, max
    Q_var = ["Max"] 
    if quartiles[-1] == 0.5: Q_len -= 1 # Median 
    
    Q = np.zeros((Q_len, zpos.shape[0]))
    Q[0] = np.max(zpos, axis=-1)
    Q[-1] = np.min(zpos, axis=-1)
    for i in range((Q_len-2)//2):
        Q_var.append("Q = " + str(1-quartiles[i]))
        Q[i+1] = np.quantile(zpos, 1-quartiles[i], axis = -1)
    if (Q_len-2)%2: 
        # Q_var.append(str(quartiles[-1]))
        Q_var.append("Median")

        Q[i+2] = np.quantile(zpos, 1-quartiles[i+1], axis = -1)
    for i in reversed(range((Q_len-2)//2)):
        Q_var.append("Q = " + str(quartiles[i]))
        Q[-i-2] = np.quantile(zpos, quartiles[i], axis = -1)
    Q_var.append("Min")
        
    return timestep, Q_var, Q    
    


def normal_buckling(sheet_dump, stretching_timestep = None):
    # --- Get data --- #
    timestep, Q_var, Q = get_normal_buckling(sheet_dump)

    # --- Plotting --- #
    # Relative to starting point
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    for i in range(Q.shape[0]):
        color = color_cycle(np.min((i, Q.shape[0]-i-1)))
        plt.plot(timestep, Q[i], color = color, label = Q_var[i])

    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("Relative sheet z-position to starting position, ($z - z_0$)", fontsize=14)
    plt.legend(fontsize = 13)

    # Relative to median
    if len(Q)%2: 
        med_idx = len(Q)//2
        plt.figure(num=1, dpi=80, facecolor='w', edgecolor='k')
        for i in range(Q.shape[0]):
            color = color_cycle(np.min((i, Q.shape[0]-i-1)))
            plt.plot(timestep, Q[i] - Q[med_idx], color = color, label = Q_var[i])

        plt.xlabel("Timestep", fontsize=14)
        plt.ylabel("Relative sheet z-position to smedian, ($z - z_{med}$)", fontsize=14)
        plt.legend(fontsize = 13)







  




if __name__ == "__main__":
    stretching_timestep = 40000
    # sheet_dump = "../area_vs_stretch/airebo_long_sheet.data";
    # sheet_dump = "../area_vs_stretch/tersoff_long_sheet.data";
    sheet_dump = "../area_vs_stretch/sheet_vacuum.data";

    # get_normal_buckling(sheet_dump)
    normal_buckling(sheet_dump, stretching_timestep = stretching_timestep)
    plt.show()