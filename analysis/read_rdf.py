### Plots the RDF 

from analysis_utils import *
from scipy.signal import argrelextrema


def read_rdf(filename):
    timestep, data = read_ave_time_vector(filename)
    dbin = np.max(data[0, 1:, 0] - data[0, :-1, 0])
    
    # plt.plot(data[100, :, 0], data[100, :, 1])
    # plt.show()
    # exit()
    
    first_peak = np.zeros(len(timestep))
    for t, step, in enumerate(timestep):
        x = data[t, :, 0]; y = data[t, :, 1]
        peaks = x[argrelextrema(y, np.greater)]
        first_peak[t] = peaks[0]   
    
    plt.plot(timestep, first_peak)
    
    
    plt.show()

if __name__ == "__main__":
    filename = "../friction_simulation/my_simulation_space/rdf.txt"
    read_rdf(filename)