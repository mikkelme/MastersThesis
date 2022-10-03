import numpy as np 
import matplotlib.pyplot as plt 


def read_friction_file(filename):
    return np.loadtxt(filename, unpack=True)

# def avg_forward(timestep, force, interval):
#     step = []
#     avg = []
#     for i in range(0, len(timestep), interval):
#         avg.append(np.mean(force[i: i+interval]))
#         step.append(np.mean(timestep[i: i+interval]) )

#     return step, avg   



def avg_forward(*args):
    product = 1
    for arg in args:
        print(arg[0])

    exit()


if __name__ == "__main__":
    interval = 1


    filenames = [
    "output_data/friction_force_stretch.txt", 
    "output_data/friction_force_nostretch.txt",
    "output_data/friction_force2xFN_stretch.txt",
    "output_data/friction_force2xFN_nostretch.txt",
    ]


    for i, filename in enumerate(filenames):
        timestep, v_F_N, f_spring_force1, f_spring_force2, f_spring_force3, f_spring_force4, c_Ff1, c_Ff2, c_Ff3, c_sheet_COM1, c_sheet_COM2, c_sheet_COM3  = read_friction_file(filename)
        timestep, c_Ff1 = avg_forward(timestep, c_Ff1)

        exit()
        fig, ax = plt.subplots(2, 2, num = i)
        fig.suptitle(filename)

        # Fx
        ax[0,0].plot(timestep, f_spring_force1, label = "spring force")
        step, avg = avg_forward(timestep, c_Ff1, interval)
        ax[0,0].plot(step, avg, label = "group/group force")
        ax[0,0].set(ylabel='$F_x$')
        # ax[0,0].label_outer()

        # Fy
        ax[0,1].plot(timestep, f_spring_force2)
        step, avg = avg_forward(timestep, c_Ff2, interval)
        ax[0,1].plot(step, avg)
        ax[0,1].set(ylabel='$F_y$')

        # |Fxy|
        ax[1,0].plot(timestep, f_spring_force4)
        # ax[1,0].plot(timestep, np.sqrt(f_spring_force1**2 + f_spring_force2**2))
        step, avg = avg_forward(timestep, np.sqrt(c_Ff1**2 + c_Ff2**2), interval)
        ax[1,0].plot(step, avg)
        ax[1,0].set(xlabel='timestep', ylabel='$|F_{xy}|$')

        # Fz
        ax[1,1].plot([],[]) # Empty to get labels right
        step, avg = avg_forward(timestep, c_Ff3, interval)
        ax[1,1].plot(step, avg)
        ax[1,1].set(xlabel='timestep', ylabel='$F_z$')

        fig.legend(loc = "lower right")
        fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)

    plt.show()






    # plt.figure(num = 3)
    # plt.plot(timestep, c_Ff3)
    plt.show()