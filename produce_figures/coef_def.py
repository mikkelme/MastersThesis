import sys
sys.path.append('../') # parent folder: MastersThesis
import matplotlib.pyplot as plt
import numpy as np
from plot_set import *


def coef_example(save = False):
    num_points = int(1e4)
    F_N = np.linspace(0, 1, num_points)
    
    # Data
    Ff_lin = 1*F_N
    Ff_lin_shift = 1*F_N + 0.5
    Ff_nonlin = 0.2*np.sin(10*F_N) + Ff_lin

    # Mu 
    mu1_lin = Ff_lin/F_N
    mu1_lin_shift = Ff_lin_shift/F_N
    mu1_nonlin = Ff_nonlin/F_N
    
    
    mu2_lin = (Ff_lin[2:] - Ff_lin[:-2])/(F_N[2:] - F_N[:-2])
    mu2_lin_shift = (Ff_lin_shift[2:] - Ff_lin_shift[:-2])/(F_N[2:] - F_N[:-2])
    mu2_nonlin = (Ff_nonlin[2:] - Ff_nonlin[:-2])/(F_N[2:] - F_N[:-2])
    
    
    
    # Arrows
    arrowprops = {'arrowstyle': '->', 'color': 'black', 'lw': 1.5}
    
    
    # --- Plot Ff vs. F_N --- #
    fig1 = plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()
    plt.plot(F_N, Ff_lin, color = color_cycle(0), linestyle = '-', label = 'Linear')  
    plt.plot(F_N, Ff_lin_shift, color = color_cycle(1), linestyle = '-', label = 'Linear + shift')  
    plt.plot(F_N, Ff_nonlin, color = color_cycle(2), linestyle = '-', label = 'Non linear')  
    
    ax.annotate('', xy=(1, 0), xytext=(0, 0), textcoords='data', arrowprops=arrowprops)
    ax.annotate('', xy=(0, 1.55), xytext=(0, 0), textcoords='data', arrowprops=arrowprops)
    # hline(ax, 0, color = 'black')
    # vline(ax, 0, color = 'black')
    
    plt.xlabel(r'$F_N$', fontsize=30)
    plt.ylabel(r'$F_{fric}$', fontsize=30)
    # plt.legend(fontsize = 18)
    plt.legend(loc='upper left', bbox_to_anchor=[0.05, 1.03], fontsize = 18)
    ax.set_facecolor("white")
    ax.set_xticks([0.0])
    ax.set_yticks([0.0])
    plt.grid(False)
    
    
    # Add slope indicator
    x = np.array([0.2, 0.4])
    xline = plt.plot(x, x*0 + x[0] + 0.5, color = 'black', linewidth = 1)
    yline = plt.plot(x*0 + x[1], x + 0.5, color = 'black', linewidth = 1)
    plt.text(np.mean(x), x[0] - 0.05 + 0.5, '1', fontsize = 20, va = 'top')
    plt.text(x[1] + 0.03, np.mean(x) + 0.5, 'a', fontsize = 20, ha = 'left')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    # --- Plot mu --- #
    
    # mu 1
    fig2 = plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()
    plt.plot(F_N, mu1_lin, color = color_cycle(0), linestyle = '-', label = r'Linear')
    plt.plot(F_N, mu1_lin_shift, color = color_cycle(1), linestyle = '-', label = r'Linear + shift')
    plt.plot(F_N, mu1_nonlin, color = color_cycle(2), linestyle = '-', label = r'Non linear')
    
    ax.annotate('', xy=(1, 0), xytext=(0, 0), textcoords='data', arrowprops=arrowprops)
    ax.annotate('', xy=(0, 5), xytext=(0, -1.3), textcoords='data', arrowprops=arrowprops)
    # hline(ax, 0, color = 'black')
    # vline(ax, 0, color = 'black')
    
    plt.xlabel(r'$F_N$', fontsize=30)
    plt.ylabel(r'$\mu$', fontsize=30)
    plt.ylim(bottom = -1.3, top = 5)
    ax.set_facecolor("white")
    ax.set_xticks([0.0])
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels([0, 'a'])
    plt.grid(False)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    
    fig3 = plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()
    plt.ylim(bottom = -1.3, top = 5)
    plt.plot(F_N[1:-1], mu2_lin, color = color_cycle(0), linestyle = '-', label = r'Linear')
    plt.plot(F_N[1:-1], mu2_lin_shift, color = color_cycle(1), linestyle = '--', label = r'Linear + shift')
    plt.plot(F_N[1:-1], mu2_nonlin, color = color_cycle(2), linestyle = '-', label = r'Non linear')
    ax.annotate('', xy=(1, 0), xytext=(0, 0), textcoords='data', arrowprops=arrowprops)
    ax.annotate('', xy=(0, 5), xytext=(0, -1.3), textcoords='data', arrowprops=arrowprops)
    # hline(ax, 0, color = 'black')
    # vline(ax, 0, color = 'black')
    
    plt.xlabel(r'$F_N$', fontsize=30)
    plt.ylabel(r'$\mu$', fontsize=30)
    ax.set_facecolor("white")
    ax.set_xticks([0.0])
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels([0, 'a'])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    
    if save:
        fig1.savefig('../article/figures/theory/fric_coef_example_a.pdf', bbox_inches='tight')
        fig2.savefig('../article/figures/theory/fric_coef_example_b.pdf', bbox_inches='tight')
        fig3.savefig('../article/figures/theory/fric_coef_example_c.pdf', bbox_inches='tight')

    
    pass

if __name__ == '__main__':
    coef_example(save = True)    
    plt.show()