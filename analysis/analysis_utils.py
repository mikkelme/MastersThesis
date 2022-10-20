import numpy as np 
import matplotlib.pyplot as plt 
from scipy import signal
import os
import pandas

import sys
sys.path.append('../') # parent folder: MastersThesis
from plot_set import *

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


def read_friction_file(filename):
    return np.loadtxt(filename, unpack=True)


def get_files_in_folder(path, ext = None):  
    filenames = []
    for file in os.listdir(path):
        if ext is None:
            filenames.append(path + file)
        else:
            if file[-4:] == ".txt":
                filenames.append(path + file)
    return filenames

def get_dirs_in_path(d):
    return [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
           

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

def decompose_wrt_drag_dir(x, y, drag_direction):
    xy_vec = np.vstack((x, y)).T

    # Directions
    dir_para = drag_direction.astype('float64')
    dir_perp = np.array((dir_para[1], -dir_para[0]))

    # Unit directions
    dir_para /= np.linalg.norm(dir_para)
    dir_perp /= np.linalg.norm(dir_perp)
    
    # Projection
    proj_para = np.dot(xy_vec, dir_para) 
    proj_perp = np.dot(xy_vec, dir_perp)
  
    return proj_para, proj_perp


def get_fricton_force(filename):
    window_length = 50
    polyorder = 5

    # Read from file
    timestep, f_move_force1, f_move_force2, c_Ff_sheet1, c_Ff_sheet2, c_Ff_sheet3, c_Ff_PB1, c_Ff_PB2, c_Ff_PB3, c_sheet_COM1, c_sheet_COM2, c_sheet_COM3 = read_friction_file(filename)
    
    
    # Find a way to get pulling direction and dt
    drag_direction = np.array((0, 1))
    dt = 0.001
    
    
    time = timestep * dt # [ps]
    
    # Organize in columns: parallel to drag, perpendicular to drag, z-axis
    move_force = np.vstack((decompose_wrt_drag_dir(f_move_force1, f_move_force2, drag_direction), np.zeros(len(f_move_force1)))).T
    Ff_sheet = np.vstack((decompose_wrt_drag_dir(c_Ff_sheet1, c_Ff_sheet2, drag_direction), c_Ff_sheet3)).T
    Ff_PB = np.vstack((decompose_wrt_drag_dir(c_Ff_PB1, c_Ff_PB2, drag_direction), c_Ff_PB3)).T
    COM_sheet = np.vstack((decompose_wrt_drag_dir(c_sheet_COM1, c_sheet_COM2, drag_direction), c_sheet_COM3)).T
    COM_sheet -= COM_sheet[0,:] # origo as reference point
    
    # # Smoothen or average
    Ff_sheet[:,0], Ff_sheet[:,1], Ff_sheet[:,2], Ff_PB[:,0], Ff_PB[:,1], Ff_PB[:,2], move_force[:,0], move_force[:,1] = savgol_filter(window_length, polyorder, Ff_sheet[:,0], Ff_sheet[:,1], Ff_sheet[:,2], Ff_PB[:,0], Ff_PB[:,1], Ff_PB[:,2], move_force[:,0], move_force[:,1])
    Ff_full_sheet = Ff_sheet + Ff_PB


    FN_full_sheet = np.mean(Ff_full_sheet[:,2])
    FN_sheet = np.mean(Ff_sheet[:,2])
    FN_PB = np.mean(Ff_PB[:,2])
    FN = np.array((FN_full_sheet, FN_sheet, FN_PB))
    
    
    max_full_sheet = Ff_full_sheet[:,0].max()
    avg_full_sheet = np.mean(Ff_full_sheet[:,0])
    
    max_sheet = Ff_sheet[:,0].max()
    avg_sheet = np.mean(Ff_sheet[:,0])
    
    max_PB = Ff_PB[:,0].max()
    avg_PB = np.mean(Ff_PB[:,0])
    
    Ff = np.array([[max_full_sheet, avg_full_sheet],
                    [max_sheet, avg_sheet],
                    [max_PB, avg_PB]])



    return Ff, FN
    # return Ff_full_sheet[:,0].max(), abs(FN_full_sheet)       
    
    
    
def organize_data(data, num_stretch): # Working title
    num_F_N = len(data)//num_stretch
    assert(len(data)%num_stretch == 0), f"Number of stretch files ({num_stretch}) does not match the total number of data points ({len(data)}))"

    # Sort by stretch pct     
    sort = np.argsort(data[:, 0])
    data = data[sort]

    # Create arrays
    stretch_pct = np.zeros(num_stretch)
    F_N = np.zeros(num_F_N)
    # Ff = np.zeros((num_stretch, num_F_N), dtype = 'object')
    Ff = np.zeros((num_stretch, num_F_N, data[0,2].shape[0],data[0,2].shape[1]))
    
    # Sort by F_N
    for i in range(num_stretch):
        interval = range(i*num_F_N, (i+1)*num_F_N)
        
        subsort = np.argsort(data[interval, 1])  
        data[interval] = data[interval][subsort]
        
        stretch_pct[i] = np.mean(data[interval,0])
        assert(np.std(data[interval,0]) < 1e-10), "Stretch_pct deviates from each other"
        
        F_N += (data[interval,1]).astype('float')
        # test = np.concatenate(data[interval,2][:], axis = 2)
        # print(test)
        # exit()
        Ff[i] = np.stack(data[interval,2])
        
    F_N /= num_stretch
    assert(np.max([np.std(data[range(i, len(data), num_F_N),1]) for i in range(num_F_N)]) < 10e-10), "F_N deviates from each other" 
    
    return stretch_pct, F_N, Ff
    
    
    

# def find_pseudo_unique(array, tol):
#     for 