
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
    threshold = [4, 3, 2, 1.5, 1.25]
    for t in threshold:
        contact_pct = np.count_nonzero(min_distances < t, axis = 1)/sheet_num_atoms
        plt.plot(timestep, contact_pct, "-o", markersize = 3, label = f"threshold = {t} Å")

    # plt.vlines(8000, 0, 0.2, linestyle = "--", color = "k", label = "Stretch begin")
    plt.legend()
    plt.xlabel("timestep")
    plt.ylabel("contact count (%)")
    plt.show()






if __name__ == "__main__":
    sheet_dump = "../area_vs_stretch/sheet.data";
    sub_dump = "../area_vs_stretch/substrate_contact.data";
    filename = "./distances.txt"

    run_calculation(sheet_dump, sub_dump, filename)
    timestep, min_distances, sheet_num_atoms = read_distance_file(filename)
    plot_contact_area(timestep, min_distances, sheet_num_atoms)