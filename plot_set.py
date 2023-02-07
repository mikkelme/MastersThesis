#Plotting settings
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("bmh")
sns.color_palette("hls", 1)

import matplotlib
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'



import numpy as np
import statsmodels.api as sm
def lin_fit(x,y):
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    res = model.fit()
    b, a = res.params
    b_err, a_err = res.bse
    return a, b, a_err, b_err

def decimals(val):
    return  int(np.ceil(-np.log10(val)))

def color_cycle(num_color):
    """ get color from matplotlib
        color cycle
        use as: color = color_cycle(3) """
    color = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return color[num_color]

def vline(ax, x, **kwargs):
    """ Make vertical line spanning whole y range
        without changing ylim of axes """
        
    ylim = ax.get_ylim()
    if hasattr(ylim[0], '__len__'):
        ylim = ylim[0]
        
    ax.vlines(x, ylim[0], ylim[1], **kwargs)
    ax.set_ylim(ylim)
    
def hline(ax, y, **kwargs):
    """ Make horisontal line spanning whole x range
        without changing xlim of axes """
        
    xlim = ax.get_xlim()
    ax.hlines(y, xlim[0], xlim[1], **kwargs)
    ax.set_xlim(xlim)
    


    
def yfill(ax, x, **kwargs):
    """ Make vertical fill spanning whole yrange
        without changing xlim or ylim of axes """
        
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.fill_between(x, ylim[0], ylim[1], **kwargs)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    

        
        


#--- Plot commands ---#
# plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
# plt.xlabel(r'$x$', fontsize=14)
# plt.ylabel(r'$y$', fontsize=14)
# plt.legend(fontsize = 13)
# plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
# plt.savefig('../article/figures/figure.pdf', bbox_inches='tight')

#--- Import from parrent folder ---#
# import sys
# sys.path.append('../') # parent folder: MastersThesis
# from plot_set import *

