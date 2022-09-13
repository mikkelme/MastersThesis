import sys
sys.path.append('../') # parent folder: MastersThesis
from plot_set import *
import numpy as np
import matplotlib.pyplot as plt
from contact_area import *

def read_stretch_file(filename):
      timestep, stretch_pct = np.loadtxt(filename, delimiter = ",", unpack = True)
      return timestep, stretch_pct


def contact_hysteresis(stretch_pct_filename, distance_filename):
    timestep_strecth, stretch_pct = read_stretch_file(stretch_pct_filename)
    timestep_distance, min_distances, sheet_num_atoms = read_distance_file(distance_filename)




    # Sync by timestep
    assert abs((timestep_strecth[1] - timestep_strecth[0]) - (timestep_distance[1] - timestep_distance[0])) < 1e-12, f"SYNC ERROR: Files {stretch_pct_filename} and {distance_filename} does not use the same output interval and cannot be synched."

    timestep_range = np.max((timestep_strecth[0], timestep_distance[0])), np.min((timestep_strecth[-1], timestep_distance[-1]))
    stretch_idx = np.argwhere(np.logical_and(timestep_range[0] <= timestep_strecth, timestep_strecth <= timestep_range[-1]))[:,0]
    distance_idx = np.argwhere(np.logical_and(timestep_range[0] <= timestep_distance, timestep_distance <= timestep_range[-1]))[:,0]


    timestep_strecth = timestep_strecth[stretch_idx]
    stretch_pct = stretch_pct[stretch_idx]
    timestep_distance = timestep_distance[distance_idx]
    min_distances = min_distances[distance_idx]

    assert len(timestep_strecth) == len(timestep_distance), f"Timestep from {stretch_pct_filename} and {distance_filename} does not match in length."
    assert np.linalg.norm(timestep_strecth - timestep_distance) < 0.01,  f"Timestep from {stretch_pct_filename} and {distance_filename} deviates in value: Not properly synced."


    # temporary fix of stretch_pct
    stretch_pct -= stretch_pct[0]


    threshold = [4.5, 4, 3.5, 3, 2.5]
    plt.figure(num=3, dpi=80, facecolor='w', edgecolor='k')

    for t in threshold:
        contact_pct = np.count_nonzero(min_distances < t, axis = 1)/sheet_num_atoms
        plt.plot(stretch_pct, contact_pct, "-o", markersize = 3, label = f"threshold = {t} Å")
    ax = plt.gca()
    ylim = ax.get_ylim()
    plt.xlabel("Stretch pct", fontsize = 14)
    plt.ylabel("contact count (%)", fontsize = 14)
    plt.legend(fontsize = 13)






if __name__ == "__main__":
    stretch_pct_filename = "../area_vs_stretch/stretch_pct_copy.txt"
    sheet_dump = "../area_vs_stretch/sheet_copy.data";
    sub_dump = "../area_vs_stretch/substrate_contact_copy.data";
    distance_filename = "./distances_copy.txt"

    # run_calculation(sheet_dump, sub_dump, distance_filename);
    contact_hysteresis(stretch_pct_filename, distance_filename)
    plt.show()
