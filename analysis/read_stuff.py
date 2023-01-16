import numpy as np
import matplotlib.pyplot as plt

from analysis_utils import *

def read_MSD(filename, create_fig = True):
    data = read_ave_time(filename)
    time = data['TimeStep']
    
    if create_fig: plt.figure(num = unique_fignum())
    MSDtol = 1.0 # For COM AVE
    plt.plot(time, data['v_MSD_com'], label = "com")
    # plt.plot(time, data['v_MSD_clean'], label = "clean")
    plt.plot(time, data['v_MSD_com_ave'], label = "com_ave")
    plt.hlines(MSDtol, data['TimeStep'][0], data['TimeStep'][-1] , linestyle = '--', color = 'black')
    plt.ylabel('MSD')
    plt.legend()
    

def read_ystress(filename, create_fig = True):
    data = read_ave_time(filename)
    time = data['TimeStep']
    stress = data['c_YS']
    
    runmax = cum_max(stress)
    YStol = 0.5*runmax
    
    if create_fig: plt.figure(num = unique_fignum())
    plt.plot(time, stress)
    plt.plot(time, YStol, linestyle = '--', color = 'black')
    plt.ylabel('$\sigma_y$')
    
    

def read_cluster(filename, create_fig = True):
    data = read_ave_time(filename)
    time = data['TimeStep']
    
    if create_fig: plt.figure(num = unique_fignum())
    plt.plot(time, data['c_Ncluster'], label = "cluster count")
    plt.ylabel('# cluster')
    plt.legend()
    


def read_CN(filename, create_fig = True):
    data = read_ave_time(filename)
    time = data['TimeStep']
    
    if create_fig: plt.figure(num = unique_fignum())
    # print(data['c_CN_ave'].max(), data['c_CN_ave'].min())
    
    sheet_atoms = 4600
    runmax = cum_max(data['c_CN_ave'])
    CNtol = (1-2/sheet_atoms)*runmax
    
    plt.plot(time, data['c_CN_ave'], label = "CN ave")
    plt.plot(data['TimeStep'], CNtol, linestyle = '--', color = 'black')
    
    plt.xlabel("Time step")
    plt.ylabel("mean (CN (cutoff = 2.2 Å))")
    
    plt.legend()
    


def read_vel(filename, create_fig = True):
    data = read_ave_time(filename)
    time = data['TimeStep']
    
    veltol = 25
    if create_fig: plt.figure(num = unique_fignum())
    
    plt.plot(time, data['v_vel_cummax_over_std'], label = "cummax/std LAMMPS")
    # plt.plot(time, data['c_vel_max'], label = "c_vel_max)")
    # plt.plot(time, data['c_ave_vel'], label = "c_ave_vel")
    # plt.plot(time, data['v_std_vel'], label = "v_std_vel")
    plt.hlines(veltol, data['TimeStep'][0], data['TimeStep'][-1] , linestyle = '--', color = 'black')
    plt.ylabel("velocity (cummax / std")
    
    
    
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
    
    
    


def read_CNP(filename, create_fig = True):
    data = read_ave_time(filename)
    time = data['TimeStep']
    
    if create_fig: plt.figure(num = unique_fignum())
    # print(data['c_CN_ave'].max(), data['c_CN_ave'].min())
    

    
    plt.plot(time, data['c_cnp_ave_1'], label = "cnp ave_1")
    plt.plot(time, data['c_cnp_ave_25'], label = "cnp ave_25")
    plt.plot(time, data['c_cnp_ave_3'], label = "cnp ave_3")
    plt.plot(time, data['c_cnp_ave_35'], label = "cnp ave_35")
    plt.plot(time, data['c_cnp_ave_4'], label = "cnp ave_4")
    
    plt.legend()
 
if __name__ == '__main__':
    
    path = '../friction_simulation/my_simulation_space/' 
    # path = '../Data/CONFIGS/honeycomb/single_run_4'
    
    
    # read_cluster(os.path.join(path,'cluster.txt'))
    # read_vel(os.path.join(path,'vel.txt'))
    # read_ystress(os.path.join(path,'YS.txt'))
    
    read_MSD(os.path.join(path, 'MSD.txt'))
    read_CN(os.path.join(path,'CN.txt'))
    read_CNP(os.path.join(path,'cnp.txt'))
    
    
    
    
    
    plt.show()