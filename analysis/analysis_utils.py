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


def read_friction_file_dict(filename):
    infile = open(filename, 'r')
    infile.readline()
    header = infile.readline().strip('#\n ').split(' ')
    infile.close()
    
    data = np.loadtxt(filename)
    return dict(zip(header, data.T))



def get_files_in_folder(path, ext = None, exclude = None): 
    """ ext: extension to include
        exclude: exclude files containing that string
    """ 
    filenames = []
    for dir in [x[0] for x in os.walk(path)]:
        for file in os.listdir(dir):
            if ext is None and exclude is None :
                filenames.append(os.path.join(dir, file))
            elif exclude is None:
                if file[-len(ext):] == ext:
                    filenames.append(os.path.join(dir, file))
            else:
                if file[-len(ext):] == ext and exclude not in file:
                    filenames.append(os.path.join(dir, file))
    return filenames

def get_dirs_in_path(d):
    return [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
           

def find_single_file(path, ext):
    file_list = [os.path.join(path, f) for f in os.listdir(path) if f[-len(ext):] == ext]
    assert(len(file_list) == 1), f"{len(file_list)} candidates for file with extenstion {ext} in {path} found:\n{file_list}."
    return file_list[0]

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
    
    

    
def organize_data(data): # Working title
    """ organize by column 0 and 1 """
    stretch_pct = np.unique(data[:,0]) 
    F_N = np.unique(data[:,1])
    
    output = []    
    for col in range(2, data.shape[1]):
        obj = data[0,col]
        # print(obj, type(obj))
        shape = (len(stretch_pct), len(F_N)) + np.shape(obj)
        # output.append(np.full(shape, np.nan, dtype = type(obj)))
        output.append(np.full(shape, np.nan, dtype = 'object'))

   
    for i, s in enumerate(stretch_pct):
        for j, fn in enumerate(F_N):
            index = np.argwhere((data[:,0] == s) & (data[:,1] == fn))
            if len(index) > 0:
                for col in range(2, data.shape[1]):
                    output[col-2][i,j] = data[index[0][0], col]
                  
    
            
    return np.array(stretch_pct, dtype = 'float'), np.array(F_N, dtype = 'float'), *output,
    
    
    
def get_color_value(value, minValue, maxValue, cmap='viridis'):
    """Get color from colormap. (From Henrik)
    Parameters
    -----------------
    :param value: Value used tpo get color from colormap
    :param minValue: Minimum value in colormap. Values below this value will saturate on the lower color of the colormap.
    :param maxValue: Maximum value in colormap. Values above this value will saturate on the upper color of the colormap.
    :returns: 4-vector containing colormap values. 
    This is useful if you are plotting data from several simulations, and want to color them based on some parameters changing between the simulations. For example, you may want the color to gradually change along a clormap as the temperature increases. 
    """
    diff = maxValue-minValue
    cmap = matplotlib.cm.get_cmap(cmap)
    rgba = cmap((value-minValue)/diff)
    return rgba


def read_histogram(filename):
    infile = open(filename, 'r')
    
    pointer = ""
    while True:
        line = infile.readline()
        if line[0] != "#":
            break
        pointer += line

    infile.seek(len(pointer), 0) # Go to start of data
    
    timestep = []
    hist = []
    for line in infile:
        step, num_bins, tot_count, missing_counts, minval, maxval = [float(word) for word in line.split(' ')]
        timestep.append(step)
        data = []
        for bin in range(int(num_bins)):
            words = infile.readline().split(' ')
            data.append([float(word) for word in words[1:]])
        hist.append(data)
            
    timestep = np.array(timestep)
    hist = np.array(hist)
    infile.close()        

    return timestep, hist






if __name__ == "__main__":
    filename = "../Data/new_potential_rip_test/cut_25stretch/cut_25stretch_chist.txt"
    # filename = "../Data/new_potential_rip_test/cut_30stretch/cut_30stretch_chist.txt"
    read_histogram(filename)
    