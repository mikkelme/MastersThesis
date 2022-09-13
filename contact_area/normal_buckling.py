import sys
sys.path.append('../') # parent folder: MastersThesis
from plot_set import *
import numpy as np
import matplotlib.pyplot as plt
from contact_area import plot_contact_area

def main(sheet_dump, stretching_timestep = None):
    sheet_infile = open(sheet_dump, "r")

    timestep = []
    zpos = []
    while True: # Timestep loop
        # --- Sheet positions --- #
        info = [sheet_infile.readline() for i in range(9)]
        if info[0] == '': break
        sheet_timestep = int(info[1].strip("\n"))
        sheet_num_atoms = int(info[3].strip("\n"))
        sheet_atom_pos = np.zeros((sheet_num_atoms, 3))
        print(f"\rtimestep = {sheet_timestep}", end = "")

        for i in range(sheet_num_atoms): # sheet atom loop
            line = sheet_infile.readline() # id x y z
            words = np.array(line.split(), dtype = float)
            sheet_atom_pos[i] = words[1:4]
      
        timestep.append(sheet_timestep)
        zpos.append(sheet_atom_pos[:,2])

    print()
    timestep = np.array(timestep)
    zpos = np.array(zpos)
    # plt.plot(timestep, np.min(zpos - np.mean(zpos[0]), axis = -1))
    # return 


    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    quartiles = [0.01, 0.05, 0.1, 0.25, 0.5]

    z0 = np.mean(zpos[0])
    zpos -= z0

    plt.plot(timestep, np.max(zpos, axis=-1), color =  color_cycle(1), label = "Max")
    for i, quartile in enumerate(quartiles):
        color = color_cycle(i+2)
        if quartile == 0.5:
            plt.plot(timestep, np.quantile(zpos, quartile, axis = -1), color = color, label = "Median")
        else:
            plt.plot(timestep, np.quantile(zpos, 1-quartile, axis = -1), color = color, label = f"Quartile = {quartile}, {1-quartile}")
            plt.plot(timestep, np.quantile(zpos, quartile, axis = -1), color = color)



    plt.plot(timestep, np.min(zpos, axis=-1), color = color_cycle(0), label = "Min")
    if stretching_timestep != None:
        plt.autoscale(False)
        plt.vlines(stretching_timestep, np.min(zpos), np.max(zpos), linestyle = "--", color = "k", label = "Stretch begin")
        plt.autoscale(True)
    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("Relative sheet z-position to starting point, ($z - z_0$)", fontsize=14)
    plt.legend(fontsize = 13)


    plt.figure(num=1, dpi=80, facecolor='w', edgecolor='k')
    quartiles = [0.01, 0.05, 0.1, 0.25, 0.5]

    z0 = np.mean(zpos[0])
    zpos -= np.quantile(zpos, 0.5, axis=-1)[:, None]
   
    plt.plot(timestep, np.max(zpos, axis=-1), color =  color_cycle(1), label = "Max")
    for i, quartile in enumerate(quartiles):
        color = color_cycle(i+2)
        if quartile == 0.5:
            plt.plot(timestep, np.quantile(zpos, quartile, axis = -1), color = color, label = "Median")
        else:
            plt.plot(timestep, np.quantile(zpos, 1-quartile, axis = -1), color = color, label = f"Quartile = {quartile}, {1-quartile}")
            plt.plot(timestep, np.quantile(zpos, quartile, axis = -1), color = color)


    plt.plot(timestep, np.min(zpos, axis=-1), color = color_cycle(0), label = "Min")
    if stretching_timestep != None:
        plt.autoscale(False)
        plt.vlines(stretching_timestep, np.min(zpos), np.max(zpos), linestyle = "--", color = "k", label = "Stretch begin")
        plt.autoscale(True)

    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("Relative sheet z-position to median, ($z - z_{med}$)", fontsize=14)
    plt.legend(fontsize = 13)






  




if __name__ == "__main__":
    stretching_timestep = 16000
    # sheet_dump = "../area_vs_stretch/airebo_long_sheet.data";
    # sheet_dump = "../area_vs_stretch/tersoff_long_sheet.data";
    sheet_dump = "../area_vs_stretch/sheet.data";


    main(sheet_dump, stretching_timestep = stretching_timestep)
    plt.show()