import sys
sys.path.append('../') # parent folder: MastersThesis
from plot_set import *
import numpy as np
import matplotlib.pyplot as plt
from contact_area import plot_contact_area
from hysteresis import *


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
    while True: # Timestep loop
        # --- Sheet positions --- #
        info = [sheet_infile.readline() for i in range(9)]
        if info[0] == '': break
        sheet_timestep = int(info[1].strip("\n"))
        # if sheet_timestep == 50000:  break
 

        sheet_num_atoms = int(info[3].strip("\n"))
        sheet_atom_pos = np.zeros((sheet_num_atoms, 3))
        print(f"\rTimestep = {sheet_timestep}", end = "")

        for i in range(sheet_num_atoms): # sheet atom loop
            line = sheet_infile.readline() # id type x y z [...]
            words = np.array(line.split(), dtype = float)
            sheet_atom_pos[i] = words[2:5] # <--- Be aware of different dumpt formats!
        
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
        
    return timestep, Q_var, Q    
    
def get_stretch_timestamps(stretch_file):
    timestep, stretch_pct, ylow_force, yhigh_force = read_stretch_file(stretch_file)
    delta_stretch = stretch_pct[1:] - stretch_pct[:-1] 
    diff = np.zeros((2, len(timestep)-2))
    diff[0] = stretch_pct[1:-1] - stretch_pct[:-2]    # backwards
    diff[1] = stretch_pct[2:] - stretch_pct[1:-1] # forward


    stretch_timestaps = np.argwhere(np.logical_and(np.min(diff, axis = 0) == 0, np.max(diff, axis = 0) != 0))[0] + 1
    # A = np.linspace(0,10, 11)**2
    # back = A[1:-1] - A[:-2]
    # forward = A[2:] - A[1:-1]

    
    # exit()
    # diff = np.array([stretch_pct[1:-1] - stretch_pct[:-2], stretch_pct[1:] - stretch_pct[:-1]]) # (backwards, forward)
    
    # test = np.argwhere(np.logical_and(diff.min() == 0, diff.max()!= 0))
    # for i in range(len(timestep)-2):
    #     print(diff[0,i], diff[1,i])


    for i in range(1, len(timestep)-2):
        diff_forward = stretch_pct[i+1] - stretch_pct[i]
        diff_backwards = stretch_pct[i] - stretch_pct[i-1]

        # print(diff[0,i-1], diff[1,i-1], diff_backwards, diff_forward)

        diff = np.array((diff_forward, diff_backwards))
        if diff.min() == 0 and diff.max() != 0:
            print(i)


        
    # constant_domains = np.argwhere(delta_stretch == 0)
    # jumps = constant_domains[1:] - constant_domains[:-1] 
    # print(constant_domains)
    exit()


def normal_buckling(sheet_dump, stretch_file = None):
    # --- Get data --- #
    timestep, Q_var, Q = get_normal_buckling(sheet_dump)
    if stretch_file != None:
        get_stretch_timestamps(stretch_file)



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
    # stretching_timestep = 40000
    # sheet_dump = "../area_vs_stretch/sheet_vacuum.data";
    sheet_dump = "../area_vs_stretch/sheet.data";
    stretch_file = "../area_vs_stretch/stretch.txt";


    normal_buckling(sheet_dump, stretch_file = stretch_file)
    plt.show()