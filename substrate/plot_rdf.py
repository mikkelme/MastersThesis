import numpy as np
import matplotlib.pyplot as plt

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from plot_set import *




def read_rdf_file(filename):
    # Read the data file
    infile = open(filename, 'r')

    data = []
    time = []

    [infile.readline() for _ in range(3)] # Skip first three lines
    for line in infile:
        timestep, num_bins = [int(word) for word in line.split(" ")]
        time.append(timestep)
        
        for i in range(num_bins):
            words = [float(word) for word in infile.readline().split(" ")]
            data.append(words[1:])
        
    time = np.array(time)
    data = np.reshape(np.array(data), (time.shape[0], num_bins, 3))
    mid =  data[0, :, 0]
    rdf = data[:,:, 1]

    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
# plt.legend(fontsize = 13)
# plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
# plt.savefig("../article/figures/figure.pdf", bbox_inches="tight")

    for i in np.linspace(1, len(time)-1, min(len(time)-1,7), dtype = int):
        plt.plot(mid, rdf[i], label = f"timestep = {time[i]}")
    
    plt.legend(fontsize = 13)
    plt.xlabel(r"Radial distance (bin center) [Å]", fontsize=14)
    plt.ylabel(r"RDF", fontsize=14)

    plt.figure(num=1, dpi=80, facecolor='w', edgecolor='k')
    rdf_change = np.linalg.norm(np.abs(rdf[1:] - rdf[:-1]), axis = 1)
    plt.plot(time[2:], rdf_change[1:])
    plt.xlabel(r"Timestep", fontsize=14)
    plt.ylabel(r"$\Delta$ RDF", fontsize=14)

    plt.show()

    # peinr()
    # bins = [0]
    # for i in range(1, len(mid)+1):
    #     bins.append(2*mid[i-1] - bins[i-1])
    # bins = np.array(bins)

    # print(len(x))
    # print(len(bins))


    # x = np.array([1, 4, 2])
    # bins = np.array([0, 4, 5, 6])
    # bins = 4
    # plt.hist(x, bins = bins)
    # plt.show()
    



# Row, bin_center, g(r), coord(r)


if __name__ == "__main__":
    filename = "rdf.txt"    
    read_rdf_file(filename)