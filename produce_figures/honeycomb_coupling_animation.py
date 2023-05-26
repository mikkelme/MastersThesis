import sys
sys.path.append('../') # parent folder: MastersThesis
import matplotlib.pyplot as plt
import numpy as np
from plot_set import *
from scipy import interpolate
from scipy.interpolate import CubicSpline
from scipy import signal



def read_coupling_file(filename):
    infile = open(filename, 'r')
    strain = []
    tension = []
    FN = []
    Ff = []
    for line in infile:
        if line[0] == '#': continue
        words = line.split(',')
        strain.append(float(words[0]))
        tension.append(float(words[1]))
        FN.append(float(words[2]))
        Ff.append(float(words[3]))

    strain = np.array(strain)
    tension = np.array(tension)
    FN = np.array(FN)
    Ff = np.array(Ff)
    
    sort = np.argsort(strain)    
    return strain[sort], tension[sort], FN[sort], Ff[sort]
    
def animation():
    
    strain, tension, FN, Ff = read_coupling_file('honeycomb_coupling.txt')

    # Working here
    strain = signal.savgol_filter(strain, 5, 2)
    plt.plot(strain)
    plt.show()
    exit()
    strain_to_FN = CubicSpline(strain, signal.savgol_filter(FN, 5, 2))
    strain_to_Ff
    
    
    plt.plot(strain, strain_to_FN(strain))
    plt.show()
    exit()

    FN_to_strain = CubicSpline(FN, signal.savgol_filter(strain, 5, 2))
    FN_to_Ff = CubicSpline(FN, signal.savgol_filter(Ff, 20, 5))




    plt.plot(FN, strain, 'o')
    plt.plot(FN, FN_to_strain(FN))
    plt.show()
    exit()
    
    
    # plt.plot(FN, savgol_strain)
    
    
    plt.plot(FN, Ff, 'o')
    plt.plot(FN, FN_to_Ff(FN))
    # plt.plot(FN, savgol_Ff)
    plt.show()


if __name__ == '__main__':
    animation()