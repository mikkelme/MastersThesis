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
    
     
    

 
if __name__ == '__main__':
    # filename = '../friction_simulation/my_simulation_space/rupture_data_test.txt'
    # filename = '../friction_simulation/my_simulation_space/MSD.txt'
    # read_MSD(filename)
    read_ystress('../friction_simulation/my_simulation_space/YS.txt')
    read_cluster('../friction_simulation/my_simulation_space/cluster.txt')
    plt.show()