import sys
sys.path.append('../') # parent folder: MastersThesis
from plot_set import *
import numpy as np
import matplotlib.pyplot as plt

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

    # plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    # zmean = np.mean(zpos, axis = -1) - np.mean(zpos[0])
    # plt.plot(timestep, zmean)

    # if stretching_timestep != None:
    #     plt.vlines(stretching_timestep, np.min(zmean), np.max(zmean), linestyle = "--", color = "k", label = "Stretch begin")
    # plt.xlabel("timestep")
    # plt.ylabel(" ...")
    # plt.legend()


    plt.figure(num=1, dpi=80, facecolor='w', edgecolor='k')
    quartiles = [0.75, 0.5, 0.25]
    z0 = np.mean(zpos[0])
    plt.plot(timestep, np.max(zpos, axis=-1) - z0, label = "Max")
    for i, quartile in enumerate(quartiles):
        plt.plot(timestep, np.quantile(zpos, quartile, axis = -1) - z0, label = f"Quartile = {quartile}")
    plt.plot(timestep, np.min(zpos, axis=-1) - z0, label = "Min")
    if stretching_timestep != None:
        plt.vlines(stretching_timestep, np.min(zmean), np.max(zmean), linestyle = "--", color = "k", label = "Stretch begin")
    plt.legend()
    plt.show()


  




if __name__ == "__main__":
    stretching_timestep = 16000
    sheet_dump = "../area_vs_stretch/sheet.data";
    main(sheet_dump, stretching_timestep = stretching_timestep)