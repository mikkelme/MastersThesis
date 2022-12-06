import numpy as np
import matplotlib.pyplot as plt

from analysis_utils import *

def read_ystress(filename):
    data = read_ave_time(filename)
    
    

    time = data['TimeStep']
    stress = data['c_y_stress']
    stress_filter = savgol_filter(5, 1, stress)[0]
    ycom = data['c_sheet_COM[2]']
    
    plt.figure(num = unique_fignum())
    plt.plot(time, stress)
    plt.plot(time, stress_filter)
    
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
    plt.show()
    
    

 
 
 
if __name__ == '__main__':
    filename = '../friction_simulation/my_simulation_space/rupture_data_test.txt'
    read_ystress(filename)