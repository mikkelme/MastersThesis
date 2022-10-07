import numpy as np 
import matplotlib.pyplot as plt 
from scipy import signal


import sys
sys.path.append('../') # parent folder: MastersThesis
from plot_set import *

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


def read_friction_file(filename):
    return np.loadtxt(filename, unpack=True)


def avg_forward(interval, *args):
    output = []
    for i, arg in enumerate(args):
        tmp = []
        for j in range(0, len(arg), interval):
            tmp.append(np.mean(arg[j: j+interval]))
        output.append(tmp)

    return *np.array(output),


def savgol_filter(window_length, polyorder, *args):
    output = []
    for i, arg in enumerate(args):        
        output.append(signal.savgol_filter(arg, window_length, polyorder))
    return *output, 


def plot_xy_time(fig, ax, x,y,time):
    """ Plot 2D x,y-plot with colorbar for time devolopment """
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

    # Set limits
    xsp = np.abs(x.max() - x.min()) * 0.1
    ysp = np.abs(y.max() - y.min()) * 0.1 
    if xsp != 0: ax.set_xlim(x.min() - xsp, x.max() + xsp)
    if ysp != 0: ax.set_ylim(y.min() - ysp, y.max() + ysp)

def plot_info(filenames):
    interval = 10
    window_length = 30
    polyorder = 3


    for i, filename in enumerate(filenames):
            timestep, v_F_N, move_force1, move_force2, c_Ff1, c_Ff2, c_Ff3, c_sheet_COM1, c_sheet_COM2, c_sheet_COM3  = read_friction_file(filename)
            # shift sign if not fixed in lammps script 
            # c_Ff1, c_Ff2, c_Ff3 = -c_Ff1, -c_Ff2, -c_Ff3

            # center COM
            c_sheet_COM1 -= c_sheet_COM1[0]
            c_sheet_COM2 -= c_sheet_COM2[0]
            
            # Smoothen or average
            c_Ff1, c_Ff2 = savgol_filter(window_length, polyorder, c_Ff1, c_Ff2)
            move_force1, move_force2 = savgol_filter(window_length, polyorder, move_force1, move_force2)
            avgstep, c_Ff3 = avg_forward(interval, timestep, c_Ff3)
            
            Fxy_norm = np.sqrt(c_Ff1**2 + c_Ff2**2)
            move_force_norm = np.sqrt(move_force1**2 + move_force2**2)
            


            fig, ax = plt.subplots(3, 2, num = i)
            fig.suptitle(filename)

            # Fx
            ax[0,0].plot(timestep, move_force1, label = "spring force", color = color_cycle(0))
            ax[0,0].plot(timestep, c_Ff1, label = "group/group force", color = color_cycle(1))
            ax[0,0].set(ylabel='$F_x$')
            ax[0,0].label_outer()

            # Fy
            ax[0,1].plot(timestep, move_force2, color = color_cycle(0))
            ax[0,1].plot(timestep, c_Ff2, color = color_cycle(1))
            ax[0,1].set(ylabel='$F_y$')

            # |Fxy|
            ax[1,0].plot(timestep, move_force_norm, color = color_cycle(0))
            ax[1,0].plot(timestep, Fxy_norm, color = color_cycle(1))
            ax[1,0].set(xlabel='timestep', ylabel='$||F_{xy}||$')

            # Fz
            ax[1,1].plot(avgstep, c_Ff3, color = color_cycle(1))
            ax[1,1].set(xlabel='timestep', ylabel='$F_z$')
            ax[1,1].set(xlabel='timestep', ylabel='$F_z$')

            # COM 
            ax[2,0].plot(timestep, c_sheet_COM1, label = "$COM_x$", color = color_cycle(2))
            ax[2,0].plot(timestep, c_sheet_COM2, label = "$COM_y$", color = color_cycle(3))
            ax[2,0].set(xlabel='timestep', ylabel='COM')

            # COM (top view) 
            plot_xy_time(fig, ax[2,1], c_sheet_COM1, c_sheet_COM2, timestep)
            ax[2,1].axis('equal')
            ax[2,1].set(xlabel='$COM_x$', ylabel='$COM_y$')


            fig.legend()
            fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)

            # --- Calculate properties ---  #
            # 1521 atoms in group full_sheet
            # 360 atoms in group PB_tot

            FN = np.mean(c_Ff3)
            # Static friction coefficient 
            mu_max = Fxy_norm.max()/abs(FN)
            mu_avg = np.mean(Fxy_norm)/abs(FN)
            print(f"mu_avg = {mu_avg:.2e}, mu_max = {mu_max:.2e}, (file = {filename}")



                   


if __name__ == "__main__":



    filenames = [
    "output_data/friction_force_SAFETY.txt",
    ]

    plot_info(filenames)
    plt.show()






  