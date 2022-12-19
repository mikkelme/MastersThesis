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
    infile = open(filename, 'r')
    infile.readline()
    header = infile.readline().strip('#\n ').split(' ')
    infile.close()
    
    data = np.loadtxt(filename)
    outdict = dict(zip(header, data.T))
    
    # Change name of keys
    xyz_to_elem = {'x':'[1]', 'y':'[2]'}
    elem_to_xyz = ['x', 'y', 'z']
    if 'f_spring_force[1]' in outdict:
        old_keys = [key for key in outdict.keys() if 'f_spring_force' in key] 
        # new_keys = ['v_move_force' + key.strip('f_spring_force') for key in old_keys ]     
        new_keys = ['v_move_force_' + elem_to_xyz[int(key.strip('f_spring_force[]'))-1] for key in old_keys ]     
        for old_key, new_key in zip(old_keys, new_keys): outdict[new_key] = outdict.pop(old_key)
        
    return outdict


def read_info_file(filename):
    dict = {}
    infile = open(filename, 'r')
    for line in infile:
        
        try:
            key, val = line.split()
        except ValueError:
            pass
            # print(f"Cannot read info line: {line}")
        try:
            val = float(val)
        except ValueError:
            pass

        dict[key] = val
    
    return dict
    


def metal_to_SI(input, key):
    # --- Convertion factors: SI -> metal --- #
    eV_over_ang_to_N = 1/6.24150907e8   # force: eV/Å -> N 
    ang_to_m = 1e-10                     # distance: Å -> m
    ps_to_s = 1e-12                     # time: ps -> s
    
    conv_dict = {"F": eV_over_ang_to_N,
                 "s": ang_to_m,
                 "t": ps_to_s}
    
    return input * conv_dict[key]

        


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

def get_dirs_in_path(d, sort = False):
    d = str(d)
    dirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    if sort: return np.sort(dirs)
    return dirs       

def find_single_file(path, ext):
    path = str(path)
    file_list = [os.path.join(path, f) for f in os.listdir(path) if f[-len(ext):] == ext]
    if len(file_list) == 0:
        raise FileNotFoundError
    if len(file_list) > 1:
        exit(f'{len(file_list)} candidates for file with extenstion {ext} in {path} found:\n{file_list}')
        # assert(len(file_list) == 1), f"{len(file_list)} candidates for file with extenstion {ext} in {path} found:\n{file_list}."
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


def analyse_friction_file(filename, mean_pct = 0.5, std_pct = None, drag_cap = None):
    # window_length = 50
    # polyorder = 5
    info = read_info_file('/'.join(filename.split('/')[:-1]) + '/info_file.txt' )
    drag_direction = np.array((info['drag_dir_x'], info['drag_dir_y']))
    dt = info['dt']
    
    data = read_friction_file(filename)  
    time = data['TimeStep'] * dt # [ps]
    VA_pos = (time - time[0]) * info['drag_speed']  # virtual atom position
    
    if drag_cap is not None:
        map = [VA_pos <= drag_cap][0]
        for key in data:
            data[key] = data[key][map]
            
    time = data['TimeStep'] * dt # [ps]
    VA_pos = (time - time[0]) * info['drag_speed']  # virtual atom position
    
     
    # Organize in columns: parallel to drag, perpendicular to drag, z-axis
    move_force = np.vstack((decompose_wrt_drag_dir(data['v_move_force_x'], data['v_move_force_y'], drag_direction), np.zeros(len(data['v_move_force_x'])))).T
    Ff_sheet = np.vstack((decompose_wrt_drag_dir(data['c_Ff_sheet[1]'], data['c_Ff_sheet[2]'], drag_direction), data['c_Ff_sheet[3]'])).T
    Ff_PB = np.vstack((decompose_wrt_drag_dir(data['c_Ff_PB[1]'], data['c_Ff_PB[2]'], drag_direction), data['c_Ff_PB[3]'])).T
    COM_sheet = np.vstack((decompose_wrt_drag_dir(data['c_sheet_COM[1]'], data['c_sheet_COM[2]'], drag_direction), data['c_sheet_COM[3]'])).T
    COM_sheet -= COM_sheet[0,:] # origo as reference point
    
    # Convert units
    move_force = metal_to_SI(move_force, 'F')*1e9
    Ff_sheet = metal_to_SI(Ff_sheet, 'F')*1e9
    Ff_PB = metal_to_SI(Ff_PB, 'F')*1e9
    
    # Smoothen
    # Ff_sheet[:,0], Ff_sheet[:,1], Ff_sheet[:,2], Ff_PB[:,0], Ff_PB[:,1], Ff_PB[:,2], move_force[:,0], move_force[:,1] = savgol_filter(window_length, polyorder, Ff_sheet[:,0], Ff_sheet[:,1], Ff_sheet[:,2], Ff_PB[:,0], Ff_PB[:,1], Ff_PB[:,2], move_force[:,0], move_force[:,1])
    Ff_full_sheet = Ff_sheet + Ff_PB
    
    FN_full_sheet = np.mean(Ff_full_sheet[:,2])
    FN_sheet = np.mean(Ff_sheet[:,2])
    FN_PB = np.mean(Ff_PB[:,2])
    FN = np.array((FN_full_sheet, FN_sheet, FN_PB))
    
    
    mean_window = int(len(time) * mean_pct)
    
    max_full_sheet = Ff_full_sheet[:,0].max()
    max_sheet = Ff_sheet[:,0].max()
    max_PB = Ff_PB[:,0].max()
    
    contact = np.vstack((data['v_full_sheet_bond_pct'], data['v_sheet_bond_pct'])).T
    
    
    if std_pct is None:
        mean_full_sheet = np.mean(Ff_full_sheet[-mean_window:,0])
        mean_sheet = np.mean(Ff_sheet[-mean_window:,0])
        mean_PB = np.mean(Ff_PB[-mean_window:,0])
        Ff_std = np.full(3, np.nan)
    
        contact_mean_full_sheet = np.mean(contact[-mean_window:, 0])
        contact_mean_sheet = np.mean(contact[-mean_window:, 1])
        contact_std = np.full(2, np.nan)
        
    else:
        std_window = int(mean_window * std_pct)
        
        mean_full_sheet, std_full_sheet = mean_cut_and_std(Ff_full_sheet[:, 0], mean_window, std_window)
        mean_sheet, std_sheet = mean_cut_and_std(Ff_sheet[:, 0], mean_window, std_window)
        mean_PB, std_PB = mean_cut_and_std(Ff_PB[:, 0], mean_window, std_window)
        
        contact_mean_full_sheet, contact_std_full_sheet = mean_cut_and_std(contact[:, 0], mean_window, std_window)
        contact_mean_sheet, contact_std_sheet = mean_cut_and_std(contact[:, 1], mean_window, std_window)    
        
        Ff_std = np.array([std_full_sheet, std_sheet, std_PB])
        contact_std = np.array([contact_std_full_sheet, contact_std_sheet])
        
    
    Ff = np.array([[max_full_sheet, mean_full_sheet],
                    [max_sheet, mean_sheet],
                    [max_PB, mean_PB]])
    contact_mean = np.array([contact_mean_full_sheet, contact_mean_sheet])
    
        
    # Calculate relative (to mean) std
    Ff_std /= Ff[:,1]   
    contact_std /= contact_mean
    
    
    
    varnames = ['time', 'move_force', 'Ff_full_sheet', 'Ff_sheet', 'Ff_PB', 'COM_sheet', 'FN', 'Ff', 'Ff_std', 'contact', 'contact_mean', 'contact_std']

    # output
    updated_data = {}
    for name in varnames:
        updated_data[name] = eval(name)
    
    return updated_data
    
    

    
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


def read_ave_time(filename):
    infile = open(filename, 'r')
    infile.readline() # Skip first comment
    var = infile.readline().strip("# \n").split(' ')
    data = np.loadtxt(filename)
    return dict([(var[i], data[:, i]) for i in range(len(var))])     
    


def read_ave_time_vector(filename):
    infile = open(filename, 'r')
    pointer = ""
    while True:
        line = infile.readline()
        if line[0] != "#":
            break
        pointer += line
    infile.seek(len(pointer), 0) # Go to start of data
    
    timestep = []
    data = []
    for line in infile:
        words = [int(w) for w in line.split()]
        timestep.append(words[0])
        Nbins = words[1]
        for i in range(Nbins):
            line = infile.readline()
            words = [float(w) for w in line.split()[1:]]
            data.append(words)
    
    timestep = np.array(timestep)
    data = np.array(data)
    
    data = data.reshape(len(timestep), Nbins, 3)
    return timestep, data
   
 


    
class interactive_plotter:
    """ Gets glitchy with multiple big figures open """
    def __init__(self, fig):
        self.cid_pick = fig.canvas.mpl_connect('button_press_event', self.pick_figure)
        self.fig = fig
        self.zoom = False
        self.ax_list = fig.axes
    
        self.x0 = np.min([ax.get_position().get_points()[0,0] for ax in self.ax_list])
        self.y0 = np.min([ax.get_position().get_points()[0,1] for ax in self.ax_list])
        self.x1 = np.max([ax.get_position().get_points()[1,1] for ax in self.ax_list])
        self.y1 = np.max([ax.get_position().get_points()[1,1] for ax in self.ax_list])
    


    def pick_figure(self, event):
        # print("clicked")
        if not self.zoom:
            if event.inaxes is not None:
            
                self.old_axes, self.old_pos = event.inaxes, event.inaxes.get_position()
                # pad = 0.1
                # event.inaxes.set_position([pad, pad, 1-2*pad, 1-2*pad]) 
                event.inaxes.set_position([self.x0, self.y0, self.x1-self.x0, self.y1-self.y0])  # [left, bottom, width, height] 
                self.toggle_axes(self.ax_list)
                self.toggle_axes([event.inaxes], visible = True)
                self.zoom = True
                
        else:
            if event.inaxes is None:
                self.toggle_axes(self.ax_list, visible = True)
                self.old_axes.set_position(self.old_pos)
                self.zoom = False
        self.fig.canvas.draw_idle()
        
    
    def toggle_axes(self, ax_list, visible = False):
        for ax in ax_list:
            ax.set_visible(visible)
            


def cum_mean(arr):
    cum_sum = np.cumsum(arr, axis = 0)
    divider = np.arange(len(cum_sum)) + 1
    return cum_sum / divider
    
def cum_std(arr, points = 100):
    start = np.argmin(np.isnan(arr))
    step = (len(arr) - 1) // points
    
    out = np.full(len(arr), np.nan)
    for i in range(start + step, len(arr), step):
        out[i] = np.std(arr[start:i+1])
    return out


# def running_mean(arr, window_len = 10):
#     assert window_len <= len(arr), "window length cannot be longer than array length."
#     assert window_len > 0, "window length must be > 0"
#     mean_window = np.ones(window_len)/window_len
    
#     left_padding = window_len//2
#     right_padding = (window_len-1)//2
#     new_arr = np.full(len(arr) + left_padding + right_padding, np.nan)
#     new_arr[left_padding or None:-right_padding or None] = arr
#     out = np.convolve(new_arr, mean_window, mode='valid')
#     # return np.convolve(arr, mean_window, mode='valid')
#     return out

def running_mean(arr, window_len = 1000):
    assert window_len <= len(arr), "window length cannot be longer than array length."
    assert window_len > 0, "window length must be > 0"
    mean_window = np.ones(window_len)/window_len
    
    
    # left_padding = window_len//2
    # right_padding = (window_len-1)//2
    
    left_padding = window_len//2 + (window_len-1)//2
    right_padding = 0
    

    # print(left_padding, right_padding)
    new_arr = np.full(len(arr) + left_padding + right_padding, np.nan)
    new_arr[left_padding or None:-right_padding or None] = arr
    
    mean = np.convolve(new_arr, mean_window, mode='valid')
    mean_sqr = np.convolve(new_arr**2, mean_window, mode='valid')
    std = np.sqrt(mean_sqr - mean**2)
    
    
    # return np.convolve(arr, mean_window, mode='valid')
    return mean, std





# def moving_std(arr, window_pct = 0.1):
#     """ Returns std of running tail window to the left 
#         corresponding x-position """
#     window = int(len(arr) * window_pct)
#     assert(window >= 10), "Window must be >= 10"
    
#     out = np.full(len(arr), np.nan)
#     for i in range(window-1, len(arr)):
#         if i%(len(arr)/10) == 0:
#             print(i/len(arr))
#         out[i] = np.std(arr[i+1-window:i+1])
#     return out
    
def cum_max(arr):
    return  np.maximum.accumulate(arr)
    
# def cumTopQuantileMax(arr, quantile):
    

def cumTopQuantileMax(arr, quantile, brute_force = False):
    
    start = int(np.ceil(1/(1-quantile)))
    topN = int((1-quantile) * len(arr[:start]))
    list_max = int((1-quantile) * len(arr)) * 1
    
    out = np.full(len(arr), np.nan)
    
    if brute_force:
        for i in range(start, len(arr)):
            # if i%(len(arr)/10) == 0:
            #     print(i/len(arr))
            topN, out[i] = TopQuantileMax(arr[:i], quantile)
            
        
        return out
    
    else:
        toplist = np.sort(arr[:start]).tolist()
        listlen = len(toplist)
        out = np.full(len(arr), np.nan)
        for i in range(start+1, len(arr)):
            # if i%(len(arr)/10) == 0:
            #     print(i/len(arr))
                

            idx = 0
            
            try:
                while arr[i-1] > toplist[idx] and idx < listlen:
                        idx += 1
            except IndexError:
                pass
        
            
            topN = int((1-quantile) * len(arr[:i]))
            toplist.insert(idx, arr[i-1])
            
            if listlen >= list_max:
                toplist.pop(0)
            else: 
                listlen += 1
            out[i] = np.mean(toplist[-topN:])
            
            
        assert np.all(np.sort(toplist) == np.array(toplist)), "toplist is not sorted correctly"
        return out
   
    
def TopQuantileMax(arr, quantile, mean = True):
    topN = int((1-quantile) * len(arr))
    topmax = arr[np.argpartition(arr, -topN)[-topN:]]
    
    if mean:
        return topN, np.mean(topmax)
    else:
        return topN, topmax
    


def add_xaxis(ax1, x, xnew, xlabel, decimals = 1):
    xlim = ax1.get_xlim()
    
    tick_loc = ax1.get_xticks()
    tick_loc = tick_loc[np.logical_and(xlim[0] < tick_loc, tick_loc < xlim[1])]
    
    sorter = np.argsort(x)
    arg_idx = np.searchsorted(x, tick_loc, sorter=sorter)
    
    map = arg_idx <= sorter[-1]
    tick_arg = sorter[arg_idx[map]]
    tick_loc = tick_loc[map]
    
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(tick_loc)
    ax2.set_xticklabels(np.round(xnew[tick_arg], decimals))
    ax2.set(xlabel=xlabel)
    
    # Position new axis behind for interactive to work
    ax1.set_zorder(ax2.get_zorder()+1)
    
    

def plot_heatmap(heat, param1, param2, title = None):
    # Heatmap
    param1_label, param1_vals = param1
    param2_label, param2_vals = param2
    
    heat = heat.astype('float')
    # param2_vals = param2_vals.astype('float')
    # param2_vals = param2_vals.astype('float')
    
    fig, ax = plt.subplots(figsize=(7, 7), num = unique_fignum())
    if title is not None:
        fig.suptitle(title)
    sns.set()
    sns.heatmap(heat.T, ax=ax,  xticklabels=np.around(param1_vals, 2),
                                yticklabels=np.around(param2_vals, 2),
                                annot = False)
    # sns.heatmap(heat, xticklabels=np.around(param1_vals, 2),
    #             yticklabels=np.around(param2_vals, 2), annot=True, ax=ax)
    
    
    ax.set_xlabel(param1_label)
    ax.set_ylabel(param2_label)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    # plt.subplots_adjust(hspace=0.3)
   

def unique_fignum():
    fignum = 0
    if fignum in plt.get_fignums():
        fignum = plt.get_fignums()[-1] + 1
    return fignum



def mean_cut_and_std(arr, mean_window, std_window):
    """ Calculate mean of last <mean_window> of array (of length: mean_window)
        and standard deviation of last <std_window> of a running mean
        with length <mean_window>. """
        
    
    assert mean_window + std_window <= len(arr), "Window error: mean_window + std_window > array length"
    assert 2 <= mean_window and mean_window <= len(arr), "Mean window must be in the interval [2, array length]"
    
        
    # Running mean of needed part for std
    end_part = mean_window + std_window 
    running_mean = np.convolve(arr[::-1][:end_part], np.ones(mean_window)/mean_window, mode='valid')
    
    # Output
    mean = running_mean[0]
    std = np.std(running_mean[:std_window])
    return mean, std
        
    
    


if __name__ == "__main__":
   
    filename = '../Data/Baseline/drag_length_size/108x113/system_108x113_Ff.txt'
     
    data = analyse_friction_file(filename)   