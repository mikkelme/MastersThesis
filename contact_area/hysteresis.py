import sys
sys.path.append('../') # parent folder: MastersThesis
from plot_set import *
import numpy as np
import matplotlib.pyplot as plt
# from contact_area import *
from normal_buckling import *


from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


def read_stretch_file(filename):
      timestep, stretch_pct, ylow_force, yhigh_force = np.loadtxt(filename, delimiter = " ", unpack = True)
      return timestep, stretch_pct, ylow_force, yhigh_force 


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


def sync(x1, x2):
    """ Sync x1 and x2 """
    assert abs((x1[1] - x1[0]) - (x2[1] - x2[0])) < 1e-12, f"SYNC ERROR: Input values does not match in interval/spacing and cannot be synched."

    x_range = np.max((x1[0], x2[0])), np.min((x1[-1], x2[-1]))
    x1_idx = np.argwhere(np.logical_and(x_range[0] <= x1, x1 <= x_range[-1]))[:,0]
    x2_idx = np.argwhere(np.logical_and(x_range[0] <= x2, x2 <= x_range[-1]))[:,0]

    x1 = x1[x1_idx]
    x2 = x2[x2_idx]

    assert len(x1) == len(x2), f"SYNC FAILED: Input values does not match in length."
    assert np.linalg.norm(x1 - x2) < 0.01,  f"SYNC FAILED: Input values deviates in value"

    return x1, x1_idx, x2_idx



def plot_hysteresis(x, y, time, title = "", num = 0):
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
 
  
    fig = plt.figure(num=num, dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()
    plt.title(title)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(time.min(), time.max())
    lc = LineCollection(segments, cmap='gist_rainbow', norm=norm)

    # Set the values used for colormapping
    lc.set_array(time)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)

    xsp = (x.max() - x.min()) * 0.1
    ysp = (y.max() - y.min()) * 0.1 
    ax.set_xlim(x.min() - xsp, x.max() + xsp)
    ax.set_ylim(y.min() - ysp, y.max() + ysp)



def buckling_hysteresis(stretch_filename):

    timestep_stretch, stretch_pct, ylow_force, yhigh_force  = read_stretch_file(stretch_filename)
    timestep_buckling, Q_var, Q = get_normal_buckling(sheet_dump, quartiles = [0.1, 0.25, 0.5])

    # Sync by timestep
    timestep, stretch_idx, buckling_idx = sync(timestep_stretch, timestep_buckling)
    stretch_pct = stretch_pct[stretch_idx]
    Q = Q[:, buckling_idx] 


    for i in range(len(Q)):
        plot_hysteresis(stretch_pct, Q[i]-Q[len(Q)//2], timestep, title = Q_var[i], num = 0)

    ax.set_ylim(-8, 8)
    # for i in range(len(Q)//2):
    #     plot_hysteresis(stretch_pct, np.abs(Q[i]-Q[-i-1]), timestep, title = i, num = i)

        




if __name__ == "__main__":
    stretch_filename = "../area_vs_stretch/stretch.txt"
    sheet_dump = "../area_vs_stretch/sheet_vacuum.data";
    # sub_dump = "../area_vs_stretch/substrate_contact_copy.data";
    # distance_filename = "./distances_copy.txt"

    # run_calculation(sheet_dump, sub_dump, distance_filename);
    # contact_hysteresis(stretch_pct_filename, distance_filename)
    buckling_hysteresis(stretch_filename)
    plt.show()
