import numpy as np 
import matplotlib.pyplot as plt 


def read_friction_file(filename):
    """ timestep, normal force (F_N), friction force (F_f) """
    timestep, F_N, F_f = np.loadtxt(filename, unpack=True)
    return timestep, F_N, F_f

if __name__ == "__main__":

    filenames = ["Ff_2ms.txt", "Ff_5ms_5ang.txt"]#, "Ff_50ms_long.txt"]
    speed = [2, 5]
    for i, filename in enumerate(filenames):
        plt.figure(num = i)
        timestep, F_N, F_f = read_friction_file(filename)
        plt.plot(timestep, F_f, label = filename)
        # plt.plot(timestep*speed[i], F_f, label = filename)
        plt.legend()
    plt.show()
    