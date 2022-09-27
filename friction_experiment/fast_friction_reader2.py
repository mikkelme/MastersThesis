import numpy as np 
import matplotlib.pyplot as plt 


def read_friction_file(filename):
    """ ... """
    timestep, v_F_N, f_spring_force1, f_spring_force2, f_spring_force3, f_spring_force4, c_Ff1, c_Ff2, c_Ff3 = np.loadtxt(filename, unpack=True)
    return timestep, v_F_N, f_spring_force1, f_spring_force2, f_spring_force3, f_spring_force4, c_Ff1, c_Ff2, c_Ff3


def avg_forward(timestep, force, interval):
    step = []
    avg = []
    for i in range(0, len(timestep), interval):
        avg.append(np.mean(force[i: i+interval]))
        step.append(np.mean(timestep[i: i+interval]) )

    return step, avg   

if __name__ == "__main__":

    filename = "friction_force.txt"
    timestep, v_F_N, f_spring_force1, f_spring_force2, f_spring_force3, f_spring_force4, c_Ff1, c_Ff2, c_Ff3 = read_friction_file(filename)
    interval = 10

    plt.figure(num = 0)
    plt.plot(timestep, f_spring_force1)
    step, avg = avg_forward(timestep, -c_Ff1, interval)
    plt.plot(step, avg)

    plt.figure(num = 1)
    plt.plot(timestep, f_spring_force2)
    step, avg = avg_forward(timestep, -c_Ff2, interval)
    plt.plot(step, avg)

    plt.show()