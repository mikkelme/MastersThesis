import sys
sys.path.append('../') # parent folder: MastersThesis
from baseline_variables import *
from kirigami_patterns import *


from scipy.interpolate import CubicSpline
from scipy.signal import argrelextrema



def plot_individual_profiles(path, save = False):
    folders = get_dirs_in_path(path)
    
    vars = ['data[\'stretch_pct\']', 'data[\'Ff\'][:, :, 0, 1]', 'data[\'F_N\']']
    axis_labels = [r'Stretch', r'$\langle F_\parallel \rangle$ [nN]', r'$F_N$ [nN]']

    for f in folders:
        config_path = find_single_file(f, '.npy')
        # name = config_path.split('/')[-1].rstrip('.npy')
        name, pattern_type = get_name(path, config_path)
        name = pattern_type + name.strip('()').replace(', ', '_')
        fig = multi_plot_compare([f], [config_path], vars, axis_labels, figsize = (7, 5), axis_scale = ['linear', 'linear'], colorbar_scale = [(0.1, 10), 'linear'], equal_axes = [False, False], rupplot = True)
        if save:
            plt.savefig(f'../article/figures/stretch_profiles/{name}.pdf', bbox_inches='tight')

        plt.close()


def get_properties(s, Ff):
    
    # Min and max Ff
    min_Ff_idx = np.argmin(Ff)
    max_Ff_idx = np.argmax(Ff)
    
    # Biggest forward drop in Ff
    loc_max = argrelextrema(Ff, np.greater_equal)[0]
    loc_min = argrelextrema(Ff, np.less_equal)[0]
    
    drop_start = 0; drop_end = 0; max_drop = 0
    for i in loc_max:
        for j in loc_min:
            if j > i: # Only look forward 
                drop = Ff[i] - Ff[j]
                if drop > max_drop:
                    drop_start = i
                    drop_end = j
                    max_drop = drop
    
    
                    
    prop = [[s[min_Ff_idx], Ff[min_Ff_idx]],
            [s[max_Ff_idx], Ff[max_Ff_idx]],
            [s[min_Ff_idx], s[max_Ff_idx], Ff[max_Ff_idx]-Ff[min_Ff_idx]],
            [s[drop_start], s[drop_end], max_drop]]
    return prop             


def get_name(path, config_path):
    # Names
    if 'popup' in path:    
        n = [int(s[-1]) for s in config_path.split('/')[-1].rstrip('.npy').split('_')] # numbers
        pattern_type = 'Tetrahedron'
        name = f'{n[1], n[2], n[0]}' # name
    elif 'honeycomb' in path: 
        n = [int(s) for s in config_path.split('/')[-1].strip('hon.npy')] # Honeycomb
        pattern_type = 'Honeycomb'
        name = f'{((1+n[0]//2), n[1], n[2], n[3])}' 
    elif 'RW' in path:
        pattern_type = 'Random walk'
        name = config_path.split('/')[-1].strip('RW.npy') # RW
        
    else:
        exit('Name definition not implemented')
        
    return name, pattern_type

def plot_profiles_together(path, save = False):
    """ Plot multiple stretch profiles (averaged over F_N) in the same plots
        using cubic spline to highlight the approximate trend """ 
    
    # Settings
    lines_per_fig = 10
    cmap = 'Paired'
    
    folders = get_dirs_in_path(path)
    sort = np.argsort(folders)
    figs = []
    # Min, Max, biggest diff, biggest drop
    # extrema = [['name', 'stretch', 1e3], 
    #            ['name', 'stretch', 0], 
    #            ['name', 'stretch_start', 'stretch_end', 0],
    #            ['name', 'stretch_start', 'stretch_end', 0]] 
    
    

    # topN = 10
    topN = 90
    names =     ['Min', 'Max', 'Max diff', 'Max drop']
    extrema =   [[], [], [], []]
    sort_cond = [lambda x: np.argsort(x),
                 lambda x: np.argsort(x)[::-1],
                 lambda x: np.argsort(np.abs(x))[::-1],
                 lambda x: np.argsort(x)[::-1]]
    for i, s in enumerate(sort):
        print(f'{i}/{len(sort)}')
        f = folders[s]
        rel_idx = i%lines_per_fig
        if rel_idx == 0:
            fig = plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
            figs.append(fig)
        
        color = get_color_value(rel_idx, 0, lines_per_fig-1, scale = 'linear', cmap = cmap)
        data = read_multi_folder(f, mean_pct = 0.5, std_pct = 0.35)
        s = data['stretch_pct']
        Ff = np.mean(data['Ff'][:, :, 0, 1], axis = 1)
        
        
        valid = ~np.isnan(Ff)
        s = s[valid]
        Ff = Ff[valid]
        prop = get_properties(s, Ff)
        
        # Scale by rupture stretch
        rup_stretch = data['rupture_stretch']
        if rup_stretch is not None:
            s /= rup_stretch
            
        # Cubic
        CS = CubicSpline(s, Ff)
        x = np.linspace(np.min(s), np.max(s), 1000)
        fit = CS(x)
        
        
        # # Polynomial fit 
        # X = np.ones((len(s), polyorder+1))
        # for p in range(1, polyorder+1):
        #     X[:, p] *= s**p
        # beta = (np.linalg.pinv(X.T @ X) @ X.T @ Ff).ravel()
        
        # # Fit 
        # x = np.linspace(np.min(s), np.max(s), 1000)
        # X = np.ones((len(x), polyorder+1))
        # for p in range(1, polyorder+1):
        #     X[:, p] *= x**p
        # fit = X @ beta

        # -- plot -- #
        config_path = find_single_file(f, '.npy')
        name, pattern_type = get_name(path, config_path)
            

            
        # Add extrema
        filename = config_path.replace(path + '/', '')
        extrema[0].append(np.array([name, filename, prop[0][0], prop[0][1]], dtype = object)) # Min Ff
        extrema[1].append(np.array([name, filename, prop[1][0], prop[1][1]], dtype = object)) # Max
        extrema[2].append(np.array([name, filename, prop[2][0], prop[2][1], prop[2][2]], dtype = object)) # Max diff
        extrema[3].append(np.array([name, filename, prop[3][0], prop[3][1], prop[3][2]], dtype = object)) # Max drop
       

        
        if rup_stretch is None:
            name += f', None'
        else:
            name += f', {rup_stretch:0.2f}'

        
        plt.scatter(s, Ff, s = 15, color = color)
        plt.plot(x, fit, linewidth = 1, color = color, label = name)
    
    
    # Show top N
    if len(extrema[0]) < topN:
        topN = len(extrema[0])
    for i, ex in enumerate(extrema):
        print(f'Extrema: {names[i]}')
        ex = np.array(ex)
        sort = sort_cond[i](ex[:, -1])[:topN]
        res = ex[sort]
        for r in res:
            for v in r:
                if isinstance(v, str):
                    print(f'{v:<20s}', end = ' ')
                else:
                    print(f'{v:.4f}', end = ' ')
            print()
        print()
        # print(ex[sort], '\n')
        
    if save: 
        for fig in figs:
            ax = fig.axes[0]
            ax.set_xlabel(r'Stretch / rupture stretch', fontsize=14)
            ax.set_ylabel(r'$\langle F_{\parallel}\rangle$ [nN]', fontsize=14)
            ax.legend(loc='upper left', bbox_to_anchor=(1.00, 1.00), ncol=1, fancybox=True, shadow=False)
            fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
            name = path.split('/')[-1]
            fig.savefig(f'../article/figures/stretch_profiles/SP_{fig.number}_{name}.pdf', bbox_inches='tight')



def patterns_and_profiles_2(save = False):
    # --- Min fric --- #
    patterns = [
                '../Data/CONFIGS/popup/pop_31', # (3,9,4)
                '../Data/CONFIGS/honeycomb/hon_6', # (2,5,1,1) 
                '../Data/CONFIGS/RW/RW12',
                ]
    
    names = ['Tetrahedron (3,9,4)', 'Honeycomb (2,5,1,1)', 'Random walk 12']
    PP(patterns, names, save = 'PP_min.pdf')

    # --- Max fric --- #
    patterns = [
                '../Data/CONFIGS/popup/pop_27', # (5,3,1)
                '../Data/CONFIGS/honeycomb/hon_12', # (2,1,1,1) 
                '../Data/CONFIGS/RW/RW96',
                ]
    
    names = ['Tetrahedron (5,3,1)', 'Honeycomb (2,1,1,1)', 'Random walk 96']
    PP(patterns, names, save = 'PP_max.pdf')
    
    # --- Max diff --- #
    patterns = [
                '../Data/CONFIGS/popup/pop_27', # (5,3,1)
                '../Data/CONFIGS/honeycomb/hon_42', # (2,1,5,3) 
                '../Data/CONFIGS/RW/RW96', 
                ]
    
    names = ['Tetrahedron (5,3,1)', 'Honeycomb (2,1,5,3)', 'Random walk 96']
    PP(patterns, names, save = 'PP_max_diff.pdf')
    
    
    # --- Max drop --- #
    patterns = [
                '../Data/CONFIGS/popup/pop_27', # (5,3,1)
                '../Data/CONFIGS/honeycomb/hon_28', # (2,3,3,3) 
                '../Data/CONFIGS/RW/RW01', 
                ]
    
    names = ['Tetrahedron (5,3,1)', 'Honeycomb (2,3,3,3)', 'Random walk 01']
    PP(patterns, names, save = 'PP_max_drop.pdf')
    
    
    
def PP(patterns, names, save = None):
    
    # Figure
    figsize = (10,6)
    width_ratios = [1, 1, 1, 0.05] # Small width for colorbar
    height_ratios = [0.8, 1]
    fig, axes = plt.subplots(2, 4, num = unique_fignum(), figsize = figsize, gridspec_kw ={'width_ratios': width_ratios, 'height_ratios': height_ratios})
    atom_radii = 0.6
        
    # Stretch profiles
    vars = ['data[\'stretch_pct\']', 'data[\'Ff\'][:, :, 0, 1]', 'data[\'F_N\']']
    axis_labels = [r'Stretch', r'$\langle F_\parallel \rangle$ [nN]', r'$F_N$ [nN]']
    multi_plot_compare(patterns, names, vars, axis_labels, figsize, axes = axes[0], add_contact = True)
    # fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2); return

    # Patterns
    for i, path in enumerate(patterns):
        ax = axes[1, i]
        
        config_path = find_single_file(path, '.npy')
        mat = np.load(config_path)
        
        plot_sheet(mat, ax, atom_radii, facecolor = 'grey', edgecolor = 'black') # Pattern   
        plot_sheet(1-mat, ax, atom_radii, facecolor = 'None', edgecolor = 'black', alpha = 0.2)  # Background
        # ax.axis('off')
        
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        ax.spines[:].set_color('white')
    
    axes[0,1].set_xlabel(axis_labels[0], fontsize = 14)
    axes[0,0].set_ylabel(axis_labels[1], fontsize = 14)
    axes[1,1].set_xlabel(r"$x$ (armchair direction)", fontsize = 14)
    axes[1,0].set_ylabel(r"$y$ (zigzag direction)", fontsize = 14)
    
    
    axes[-1, -1].axis('off')
    
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save is not None:
        fig.savefig(f'../article/figures/stretch_profiles/{save}', bbox_inches='tight')
    



def patterns_and_profiles(save = False):
    
    patterns = ['../Data/CONFIGS/popup/pop_31', # (3,9,4)
                '../Data/CONFIGS/popup/pop_27', # (5,3,1)
                '../Data/CONFIGS/honeycomb/hon_6', # (2,5,1,1) 
                '../Data/CONFIGS/honeycomb/hon_12', # (2,1,1,1) 
                '../Data/CONFIGS/honeycomb/hon_42', # (2,1,5,3) 
                '../Data/CONFIGS/honeycomb/hon_28', # (2,3,3,3) 
                '../Data/CONFIGS/RW/RW01', 
                '../Data/CONFIGS/RW/RW12', 
                '../Data/CONFIGS/RW/RW96', 
                ]
    
    patterns = ['../Data/CONFIGS/RW/RW01', 
                '../Data/CONFIGS/RW/RW12', 
                '../Data/CONFIGS/RW/RW96' ]
    
    
    for path in patterns:
        data = read_multi_folder(path, mean_pct = 0.5, std_pct = 0.35)
        config_path = find_single_file(path, '.npy')
        
        # --- Figure --- #
        figsize = (10,5)
        width_ratios = [1, 1, 0.05] # Small width for colorbar
        fig, axes = plt.subplots(1, 3, num = unique_fignum(), figsize = figsize, gridspec_kw ={'width_ratios': width_ratios})
        atom_radii = 0.6
        center_radii = 0.2
        
        # --- Pattern --- #
        mat = np.load(config_path)
        plot_sheet(mat, axes[0], atom_radii, facecolor = 'grey', edgecolor = 'black') # Pattern   
        plot_sheet(1-mat, axes[0], atom_radii, facecolor = 'None', edgecolor = 'black', alpha = 0.2)  # Background
        # plot_center_coordinates(np.shape(mat), axes[0], center_radii, facecolor = blue, edgecolor = None) # Center elements
        
        name, pattern_type = get_name(path, config_path)
        axes[0].set_title(f'{pattern_type} {name}')
        axes[0].set_xlabel(r"$x$ (armchair direction)", fontsize = 14)
        axes[0].set_ylabel(r"$y$ (zigzag direction)", fontsize = 14)
        axes[0].grid(False)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_facecolor("white")
        
        # axes[0].axis('off')
        
        
        # --- Stretch profile --- #
        vars = ['data[\'stretch_pct\']', 'data[\'Ff\'][:, :, 0, 1]', 'data[\'F_N\']']
        axis_labels = [r'Stretch', r'$\langle F_\parallel \rangle$ [nN]', r'$F_N$ [nN]']
        multi_plot_compare([path], [config_path], vars, axis_labels,  axes = axes[1:], axis_scale = ['linear', 'linear'], colorbar_scale = [(0.1, 10), 'linear'], equal_axes = [False, False], rupplot = True)
        axes[1].set_title('')
        axes[1].set_xlabel(axis_labels[0], fontsize = 14)
        axes[1].set_ylabel(axis_labels[1], fontsize = 14)
        
        handles, labels = axes[-2].get_legend_handles_labels()
        axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True, shadow=False, fontsize = 13)
            


        # --- Figure refinement --- #
        fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        
        
        if save:
            name = path.split('/')[-1]
            fig.savefig(f'../article/figures/stretch_profiles/PP_{name}.pdf', bbox_inches='tight')
            
        
    

if __name__ == "__main__":
    # path = '../Data/CONFIGS/popup'
    # path = '../Data/CONFIGS/honeycomb'
    path = '../Data/CONFIGS/RW'
    
    # patterns_and_profiles(save = False)
    patterns_and_profiles_2(save = False)
    plt.show()
    
    
    
    
    # plot_individual_profiles(path, save = False)
    # plot_profiles_together(path, save = False)
    # plt.show()    
    
