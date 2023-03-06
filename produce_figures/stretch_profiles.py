import sys
sys.path.append('../') # parent folder: MastersThesis
from baseline_variables import *
from scipy.interpolate import CubicSpline

def plot_individual_profiles(path, save = False):
    folders = get_dirs_in_path(path)
    
    vars = ['data[\'stretch_pct\']', 'data[\'Ff\'][:, :, 0, 1]', 'data[\'F_N\']']
    axis_labels = [r'Stretch', r'$\langle F_\parallel \rangle$ [nN]', r'$F_N$ [nN]']

    for f in folders:
        config_path = find_single_file(f, '.npy')
        name = config_path.split('/')[-1].rstrip('.npy')
        fig = multi_plot_compare([f], [config_path], vars, axis_labels, figsize = (7, 5), axis_scale = ['linear', 'linear'], colorbar_scale = [(0.1, 10), 'linear'], equal_axes = [False, False], rupplot = True)
        if save:
            plt.savefig(f'../article/figures/stretch_profiles/{name}.pdf', bbox_inches='tight')

        plt.close()

def plot_profiles_together(path, save = False):
    """ Plot multiple stretch profiles (averaged over F_N) in the same plots
        using cubic spline to highlight the approximate trend """ 
    
    # Settings
    # polyorder = 10
    lines_per_fig = 10
    cmap = 'gist_rainbow'
    
    folders = get_dirs_in_path(path)    
    for i, f in enumerate(folders):
        rel_idx = i%lines_per_fig
        if rel_idx == 0:
            plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
        
        color = get_color_value(rel_idx, 0, lines_per_fig-1, scale = 'linear', cmap = cmap)
        data = read_multi_folder(f, mean_pct = 0.5, std_pct = 0.35)
        s = data['stretch_pct']
        Ff = np.mean(data['Ff'][:, :, 0, 1], axis = 1)
        
        valid = ~np.isnan(Ff)
        s = s[valid]
        Ff = Ff[valid]
        
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
        name = config_path.split('/')[-1].rstrip('.npy')

        plt.scatter(s, Ff, s = 15, color = color)
        plt.plot(x, fit, linewidth = 1, color = color, label = name)
        
        if i%lines_per_fig == lines_per_fig-1:
            if save: 
                plt.xlabel(r'Stretch', fontsize=14)
                plt.ylabel(r'$\langle F_{\parallel}\rangle$ [nN]', fontsize=14)
                plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
                plt.legend(fontsize = 13)
                if save:
                    name = path.split('/')[-1]
                    num = i//lines_per_fig
                    plt.savefig(f'../article/figures/stretch_profiles/CS_profiles_{num}_{name}.pdf', bbox_inches='tight')


        
    

if __name__ == "__main__":
    path = '../Data/CONFIGS/honeycomb'
    # path = '../Data/CONFIGS/popup'
    # plot_individual_profiles(path, save = False)
    # plot_profiles_together(path, save = True)
    # plt.show()    
    
