import numpy as np
import matplotlib.pyplot as plt

from analysis_utils import *

def read_ystress(filename):
    data = read_ave_time(filename)
    

    time = data['TimeStep']
    stress = data['c_YS']
    
    plt.figure(num = unique_fignum())
    plt.plot(time, stress)
    
    
    # plt.figure(num = unique_fignum())
    # vel = stress_filter[1:] - stress_filter[:-1]
    # plt.plot(time[1:], vel)
    # plt.plot(time[1:], savgol_filter(20, 1, vel)[0])
    
    # plt.figure(num = unique_fignum())
    # plt.plot(time, ycom)
    
    # plt.figure(num = unique_fignum())
    # ycom_speed = np.abs(ycom[1:] - ycom[:-1])
    # plt.plot(time[1:],ycom_speed, label = "homemade")
    # plt.plot(time, data['v_sheet_VCM'], label = "in lammps")
    # plt.legend()
    
    

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
    
    

def read_cluster(filename):
    data = read_ave_time(filename)
    
    time = data['TimeStep']
    # print(data.keys())
    # exit()
    plt.figure(num = unique_fignum())
    plt.plot(time, data['c_Ncluster'], label = "cluster count")
  
    
    plt.legend()
    


def read_CN(filename):
    data = read_ave_time(filename)
    
    time = data['TimeStep']
    plt.figure(num = unique_fignum())
    print(data['c_CN_ave'].max(), data['c_CN_ave'].min())
    plt.plot(time, data['c_CN_ave'], label = "CN ave")
    plt.xlabel("Time step")
    plt.ylabel("mean (CN (cutoff = 2.2 Å))")
    
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
    
    read_ystress('../friction_simulation/my_simulation_space/YS.txt')
    # read_cluster('../friction_simulation/my_simulation_space/cluster.txt')
    # read_CN('../friction_simulation/my_simulation_space/CN.txt')
    plt.show()