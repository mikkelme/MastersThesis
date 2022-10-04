import numpy as np 
import matplotlib.pyplot as plt 
from scipy import signal


def read_friction_file(filename):
    return np.loadtxt(filename, unpack=True)


def avg_forward(interval, *args):
    output = []
    for i, arg in enumerate(args):
        tmp = []
        for j in range(0, len(timestep), interval):
            tmp.append(np.mean(arg[j: j+interval]))
        output.append(tmp)

    return *np.array(output),


def savgol_filter(window_length, polyorder, *args):
    output = []
    for i, arg in enumerate(args):
        
        output.append(signal.savgol_filter(arg, window_length, polyorder))
        print(output[i][5])
    return *output, 


if __name__ == "__main__":
    interval = 20
    window_length = 100
    polyorder = 5


    filenames = [
    # "output_data/friction_force_stretch.txt", 
    # "output_data/friction_force_nostretch.txt",
    # "output_data/friction_force2xFN_stretch.txt",
    # "output_data/friction_force2xFN_nostretch.txt",
    "output_data/friction_force_6xFN_long.txt"
    ]

    for i, filename in enumerate(filenames):
        timestep, v_F_N, f_spring_force1, f_spring_force2, f_spring_force3, f_spring_force4, c_Ff1, c_Ff2, c_Ff3, c_sheet_COM1, c_sheet_COM2, c_sheet_COM3  = read_friction_file(filename)
        
        # shift sign if not fixed in lammps script 
        c_Ff1, c_Ff2, c_Ff3 = -c_Ff1, -c_Ff2, -c_Ff3

        # center COM
        c_sheet_COM1 -= c_sheet_COM1[0]
        c_sheet_COM2 -= c_sheet_COM2[0]
        c_Ff1, c_Ff2 = savgol_filter(window_length, polyorder, c_Ff1, c_Ff2)
        
        avgstep, c_Ff3 = avg_forward(interval, timestep, c_Ff3)

        fig, ax = plt.subplots(3, 2, num = i)
        fig.suptitle(filename)

        # Fx
        ax[0,0].plot(timestep, f_spring_force1, label = "spring force")
        ax[0,0].plot(timestep, c_Ff1, label = "group/group force")
        ax[0,0].set(ylabel='$F_x$')
        ax[0,0].label_outer()

        # Fy
        ax[0,1].plot(timestep, f_spring_force2)
        ax[0,1].plot(timestep, c_Ff2)
        ax[0,1].set(ylabel='$F_y$')

        # |Fxy|
        ax[1,0].plot(timestep, f_spring_force4)
        ax[1,0].plot(timestep, np.sqrt(c_Ff1**2 + c_Ff2**2))
        ax[1,0].set(xlabel='timestep', ylabel='$|F_{xy}|$')

        # Fz
        ax[1,1].plot([],[]) # Empty to get labels right
        ax[1,1].plot(avgstep, c_Ff3)
        ax[1,1].set(xlabel='timestep', ylabel='$F_z$')
        ax[1,1].set(xlabel='timestep', ylabel='$F_z$')

        # COM 
        ax[2,0].plot([],[]) # Empty to get labels right
        ax[2,0].plot([],[]) # Empty to get labels right
        ax[2,0].plot(timestep, c_sheet_COM1, label = "$COM_x$")
        ax[2,0].plot(timestep, c_sheet_COM2, label = "$COM_y$")
        ax[2,0].set(xlabel='timestep', ylabel='COM')

        # COM (top view) 
        ax[2,1].plot([],[]) # Empty to get labels right
        ax[2,1].plot([],[]) # Empty to get labels right
        ax[2,1].plot([],[]) # Empty to get labels right
        ax[2,1].plot([],[]) # Empty to get labels right
        ax[2,1].plot(c_sheet_COM1, c_sheet_COM2, label = "$COM_xy$")
        ax[2,1].set(xlabel='$COM_x$', ylabel='$COM_x$')




        fig.legend(loc = "lower right")
        fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)

        # # Fx
        # ax[0,0].plot(timestep, f_spring_force1, label = "spring force")
        # step, avg = avg_forward(timestep, c_Ff1, interval)
        # ax[0,0].plot(step, avg, label = "group/group force")
        # ax[0,0].set(ylabel='$F_x$')
        # # ax[0,0].label_outer()

        # # Fy
        # ax[0,1].plot(timestep, f_spring_force2)
        # step, avg = avg_forward(timestep, c_Ff2, interval)
        # ax[0,1].plot(step, avg)
        # ax[0,1].set(ylabel='$F_y$')

        # # |Fxy|
        # ax[1,0].plot(timestep, f_spring_force4)
        # # ax[1,0].plot(timestep, np.sqrt(f_spring_force1**2 + f_spring_force2**2))
        # step, avg = avg_forward(timestep, np.sqrt(c_Ff1**2 + c_Ff2**2), interval)
        # ax[1,0].plot(step, avg)
        # ax[1,0].set(xlabel='timestep', ylabel='$|F_{xy}|$')

        # # Fz
        # ax[1,1].plot([],[]) # Empty to get labels right
        # step, avg = avg_forward(timestep, c_Ff3, interval)
        # ax[1,1].plot(step, avg)
        # ax[1,1].set(xlabel='timestep', ylabel='$F_z$')

        # fig.legend(loc = "lower right")
        # fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)

    plt.show()






  