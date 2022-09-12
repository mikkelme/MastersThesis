
import sys
sys.path.append('../') # parent folder: MastersThesis
from plot_set import *
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sub

def run_calculation(sheet_dump, sub_dump, filename, script = "./dis_calc.out"):
    # sub.run(script, shell=True)
    sub.run([script, sheet_dump, sub_dump, filename])



def read_distance_file(filename):
    timestep = []
    min_distances = []
    infile = open(filename, 'r')
    while True: # Timestep loop
        info = [infile.readline() for i in range(5)]
        if info[0] == '': break # EOF

        timestep.append(int(info[1].strip("\n")))
        sheet_num_atoms = int(info[3].strip("\n"))

        min_distance = np.zeros(sheet_num_atoms)
        for i in range(sheet_num_atoms):
            line = infile.readline() # idx, min dis
            min_distance[i] = float(line.split()[-1])
        min_distances.append(min_distance)

    return np.array(timestep), np.array(min_distances), sheet_num_atoms





def plot_contact_area(timestep, min_distances, sheet_num_atoms):
    # plt.plot(timestep, np.min(min_distances - np.mean(min_distances[0]), axis = -1))
    plt.plot(min_distances[0])
    return 


    threshold = [4.5, 4, 3.5, 3, 2.5]
    plt.figure(num=1, dpi=80, facecolor='w', edgecolor='k')
    for t in threshold:
        contact_pct = np.count_nonzero(min_distances < t, axis = 1)/sheet_num_atoms
        plt.plot(timestep, contact_pct, "-o", markersize = 3, label = f"threshold = {t} Å")

    plt.vlines(16000, 0, 0.2, linestyle = "--", color = "k", label = "Stretch begin")
    plt.legend()
    plt.xlabel("timestep", fontsize = 14)
    plt.ylabel("contact count (%)", fontsize = 14)
    plt.legend(fontsize = 13)






if __name__ == "__main__":
    sheet_dump = "../area_vs_stretch/airebo_long_sheet.data";
    sub_dump = "../area_vs_stretch/airebo_long_substrate_contact.data";
    filename = "./distances.txt"

    run_calculation(sheet_dump, sub_dump, filename); exit();
    timestep, min_distances, sheet_num_atoms = read_distance_file(filename)
    plot_contact_area(timestep, min_distances, sheet_num_atoms)
    plt.show()
