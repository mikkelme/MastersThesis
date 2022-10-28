import numpy as np
import matplotlib.pyplot as plt

def read_max_stress(filename, num = 0):
    infile = open(filename, 'r')
    
    line = infile.readline()
    while line[0] == "#":
        line = infile.readline()
    words = line.split(' ')
    step = [float(words[0])]
    max_stress = [[float(words[i]) for i in range(1, len(words))]]
    for line in infile:
        try:
            words = infile.readline().split(' ')
            step.append(float(words[0]))
            max_stress.append([float(words[i]) for i in range(1, len(words))])

        except ValueError:
            break
       
  
    step = np.array(step)
    max_stress = np.array(max_stress)
    
    plt.figure(num = num)
    markers = [4500, 5000, 6600, 7100]
    # markers = []
    for marker in markers:
        plt.vlines(markers, np.min(max_stress), np.max(max_stress), linestyle = "--", color = 'k')
    
    # plt.plot(step, max_stress[:,0], label = "0")
    # plt.plot(step, max_stress[:,1], label = "1")
    plt.plot(step, max_stress[:,2], label = "2")
    plt.legend()
    
if __name__ == "__main__":
    # read_max_stress('max_stress_rerun.txt')
    read_max_stress('max_stress.txt', num = 1)
    plt.show()
    
    