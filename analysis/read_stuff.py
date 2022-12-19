import numpy as np
import matplotlib.pyplot as plt

from analysis_utils import *

def read_MSD(filename):
    data = read_ave_time(filename)
    
    time = data['TimeStep']
    print(data.keys())
    
    plt.figure(num = unique_fignum())
    plt.plot(time, data['c_MSD_clean[2]'], label = "clean")
    plt.plot(time, data['c_MSD_com[2]'], label = "com")
    plt.plot(time, data['c_MSD_com_ave[2]'], label = "com + ave")
    plt.plot(time, data['c_disp_ave'], label = "disp ave")
    plt.plot(time, data['c_disp_max'], label = "disp max")
    
    plt.legend()
    

def read_ystress(filename):
    data = read_ave_time(filename)
    
    time = data['TimeStep']
    stress = data['c_YS']
    
    runmax = cum_max(stress)
    YStol = 0.95*runmax
    
    plt.figure(num = unique_fignum())
    plt.plot(time, stress)
    plt.plot(time, YStol, linestyle = '--', color = 'black')
    

    
    

def read_cluster(filename):
    data = read_ave_time(filename)
    time = data['TimeStep']
    plt.figure(num = unique_fignum())
    plt.plot(time, data['c_Ncluster'], label = "cluster count")
    plt.legend()
    


def read_CN(filename):
    data = read_ave_time(filename)
    
    time = data['TimeStep']
    plt.figure(num = unique_fignum())
    # print(data['c_CN_ave'].max(), data['c_CN_ave'].min())
    
    sheet_atoms = 4600
    runmax = cum_max(data['c_CN_ave'])
    CNtol = (1-2/sheet_atoms)*runmax
    
    plt.plot(time, data['c_CN_ave'], label = "CN ave")
    plt.plot(data['TimeStep'], CNtol, linestyle = '--', color = 'black')
    
    plt.xlabel("Time step")
    plt.ylabel("mean (CN (cutoff = 2.2 Å))")
    
    plt.legend()
    


def read_vel(filename):
    data = read_ave_time(filename)
    time = data['TimeStep']
    
    veltol = 25
    
    plt.figure(num = unique_fignum())
    plt.plot(time, data['v_vel_max_over_std'], label = "max/std LAMMPS")
    plt.hlines(veltol, data['TimeStep'][0], data['TimeStep'][-1] , linestyle = '--', color = 'black')
    
    
    
    
    # plt.plot(time, np.sqrt(data['v_var_vel']), label = "std")
    
    # plt.plot(time, cum_max(data['c_vel_max']), label = "cummax c_vel_max")
    # plt.plot(time, data['v_vel_max'], label = "v_vel_max")
    
    
    # plt.plot(time, cum_max(data['c_max_vel'])/np.mean(data['v_var_vel']), label = "max/std PYTHON")
    # plt.plot(time, cum_max(data['c_max_vel'])/np.min(data['v_var_vel']), label = "max/ min std PYTHON")
    # plt.xlabel("Time step")
    # plt.ylabel("Velocity [Å/ps]")
    
    # data = read_ave_time('../friction_simulation/my_simulation_space/vel2.txt')
    # plt.plot(data['TimeStep'], cum_max(data['c_vel_max'])/np.mean(data['v_std_vel']), label = "max/std PYTHON")
    # c_vel_max v_std_vel 
    
    plt.legend()
    
    
    

def read_rdf(filename):
    timestep, data = read_ave_time_vector(filename)
    
    idx = 0
    r = data[idx, :, 0]
    gr = data[idx, :, 1]
    group_coord = data[idx, :, 2]
    
    plt.figure(num = unique_fignum())
    plt.plot(r, gr, label = "cluster count")
    plt.xlabel("r [Å]")
    plt.ylabel("g(r)")
    
    plt.figure(num = unique_fignum())
    plt.plot(r, group_coord, label = "cluster count")
    plt.xlabel("r [Å]")
    plt.ylabel("coord(r)")
    # plt.legend()
 
if __name__ == '__main__':
    # filename = '../friction_simulation/my_simulation_space/rupture_data_test.txt'
    # filename = '../friction_simulation/my_simulation_space/MSD.txt'
    # read_MSD(filename)
    

    # read_rdf('../friction_simulation/my_simulation_space/rdf.txt')
    
    # read_ystress('../friction_simulation/my_simulation_space/YS.txt')
    # read_cluster('../friction_simulation/my_simulation_space/cluster.txt')
    # read_CN('../friction_simulation/my_simulation_space/CN.txt')
    read_vel('../friction_simulation/my_simulation_space/vel.txt')
    
    plt.show()