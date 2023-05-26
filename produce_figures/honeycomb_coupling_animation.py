import sys
sys.path.append('../') # parent folder: MastersThesis
import matplotlib.pyplot as plt
import numpy as np
from plot_set import *
from analysis.analysis_utils import *
from scipy import interpolate
from scipy.interpolate import CubicSpline
from scipy import signal
import os


def read_coupling_file(filename):
    infile = open(filename, 'r')
    strain = []
    tension = []
    FN = []
    Ff = []
    for line in infile:
        if line[0] == '#': continue
        words = line.split(',')
        strain.append(float(words[0]))
        tension.append(float(words[1]))
        FN.append(float(words[2]))
        Ff.append(float(words[3]))

    strain = np.array(strain)
    tension = np.array(tension)
    FN = np.array(FN)
    Ff = np.array(Ff)
    
    sort = np.argsort(strain)    
    return strain[sort], tension[sort], FN[sort], Ff[sort]


# def get_image(idx):
#     folder = '../presentation/figures/hon_stretch'
#     images = np.sort(os.listdir(folder))
#     return os.path.join(folder, images[idx])

def idx_to_strain(idx):
    start = 15 + 2*5 # ps
    strain_rate = 0.001 # 1/ps
    dump_freq = 10000
    dt = 1e-3 # ps
    
    t = np.max((idx * dump_freq * dt - start, 0))
    strain = t * strain_rate
    return strain
    

def animation():
    strain, tension, FN, Ff = read_coupling_file('honeycomb_coupling.txt')
    # strain_to_FN = CubicSpline(strain, signal.savgol_filter(FN, 10, 4))
    # strain_to_Ff = CubicSpline(strain, signal.savgol_filter(Ff, 5, 2))
    strain_to_FN = interpolate.interp1d(strain, FN)
    strain_to_Ff = interpolate.interp1d(strain, Ff)
    
    image_folder = '../presentation/figures/hon_stretch'
    images = np.sort(os.listdir(image_folder))
    


    # s = np.linspace(strain[0], strain[-1], int(1e4))
    # # plt.plot(strain_to_FN(s), s, color = color_cycle(1))
    # # plt.plot(FN, strain, 'o', color = color_cycle(1))
    
    # plt.plot(strain_to_FN(s), strain_to_Ff(s), color = color_cycle(0))
    # plt.plot(FN, Ff, 'o', color = color_cycle(1))

    # plt.show()
    # exit()
    
    save_folder = 'hon_stretch_anim'
    for idx, image in enumerate(images):
        s = idx_to_strain(idx)
        if s > strain[-1]:
            break
        print(f'idx = {idx}/{len(images)}, strain = {s}')
        
        # --- Plotting --- #
        fig = plt.figure(figsize = (8,7))
        gs = fig.add_gridspec(2,2, height_ratios=[1,2])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        scatter = {'zorder': 2, 'color': color_cycle(1), 'edgecolor': 'black'}

        # Left plot
        ax1.plot(strain_to_FN(strain), strain, color = color_cycle(0))
        ax1.scatter(strain_to_FN(s), s, **scatter)
        add_xaxis(ax1, x = strain_to_FN(strain), xnew = strain_to_FN(strain)*6, xlabel = 'Tension [nN]', decimals = 1, fontsize = 14)
        ax1.set_xlabel(r'$F_N$ [nN]', fontsize = 14)
        ax1.set_ylabel('Strain', fontsize = 14)


        # Right plot
        ax2.plot(strain_to_FN(strain), strain_to_Ff(strain), color = color_cycle(0))
        ax2.scatter(strain_to_FN(s), strain_to_Ff(s), **scatter)
        ax2.set_xlabel(r'$F_N$ [nN]', fontsize = 14)
        ax2.set_ylabel(r'$\langle F_\parallel \rangle$ [nN]', fontsize = 14)
        

        # I mage
        ax3.imshow(plt.imread(os.path.join(image_folder, image)))
        ax3.grid(False)
        ax3.set_xticks([]) 
        ax3.set_yticks([]) 
        # ax3.axis('off')
        ax3.set_xlabel(f'Strain = {s:0.2f}', fontsize = 14)
        
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        
        plt.savefig(f'../presentation/figures/hon_stretch_anim/frame{idx}.png', bbox_inches='tight')
        plt.close()
    

if __name__ == '__main__':
    animation()
    