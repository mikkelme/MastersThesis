import sys
sys.path.append('../') # parent folder: MastersThesis

from produce_figures.baseline_variables import *
from analysis.analysis_utils import *

    
def read_multi_coupling(folder, mean_pct = 0.5, std_pct = 0.35, stretch_lim = [None, None],  FN_lim = [None, None]):
    """ Read multi folder
    
    Expected data structure:
    
    Header_name
        info_file.txt
        (rupture_test.txt)
        |--> stretch_{TimeStep}_folder
                      : 
        |--> stretch_{TimeStep}_folder   
            |--> job0
                :
            |--> jobN
                |--> info_file.txt
                |--> system_drag_Ff.txt

    Args:
        folder (_type_): _description_
        mean_pct (float, optional): _description_. Defaults to 0.5.
        std_pct (float, optional): _description_. Defaults to 0.2.
        stretch_lim (list, optional): _description_. Defaults to [None, None].
        FN_lim (list, optional): _description_. Defaults to [None, None].

    Returns:
        _type_: _description_
    """    
    
    
    # Settings
    info_file = 'info_file.txt'
    rupture_file = 'rupture_test.txt'
    friction_ext = 'Ff.txt'
    
    
    if rupture_file in os.listdir(folder):
        rupture_info = read_info_file(os.path.join(folder, rupture_file))
        if rupture_info['is_ruptured']:
            rupture_stretch = rupture_info['rupture_stretch']
        else:
            rupture_stretch = None
    else:
        rupture_stretch = None


    # Make list for data 
    data = [] # Measurements
    rupture = [] # Ruptures
    
    dirs = get_dirs_in_path(folder, sort = True)
    # Loop through sub folders corresponding to stretch and load, format: stretch_{TimeStep}_folder
    for i, sub in enumerate(dirs): 
        # Expects only on folder from here
                
        # Loop through F_N folders, format: job{j}
        for j, subsub in enumerate(get_dirs_in_path(sub, sort = True)):
            print(f"\r ({i+1}/{len(dirs)}) | {subsub} ", end = " ")
            
            try: # If info file exist
                info_dict = read_info_file(os.path.join(subsub, info_file))
                
                if 'is_ruptured' in info_dict:
                    is_ruptured = info_dict['is_ruptured']
                    
                else:
                    print("Sim not done")
                    continue
                

                # Get data
                friction_file = find_single_file(subsub, ext = friction_ext)    
                raw_data = read_friction_file(friction_file)   
         
                
                stretch_pct = info_dict['SMAX']
                F_N = -metal_to_SI(np.mean(raw_data['c_Ff_sheet[3]'] + raw_data['c_Ff_PB[3]']), 'F')*1e9
                
                rupture.append((stretch_pct, F_N, is_ruptured, subsub))  
                
                if not is_ruptured:
                    _, fricData = analyse_friction_file(friction_file, mean_pct, std_pct)
                    if 'v_stretch_pct' in raw_data:
                        mean_stretch = np.mean(raw_data['v_stretch_pct']) 
                        std_stretch = np.std(raw_data['v_stretch_pct'])
                        data.append((stretch_pct, F_N, fricData['Ff'], fricData['Ff_std'], fricData['contact_mean'], fricData['contact_std'], mean_stretch, std_stretch))  
                    else:
                        data.append((stretch_pct, F_N, fricData['Ff'], fricData['Ff_std'], fricData['contact_mean'], fricData['contact_std']))  
                else:
                    if 'v_stretch_pct' in raw_data:
                        mean_stretch = np.mean(raw_data['v_stretch_pct']) 
                        std_stretch = np.std(raw_data['v_stretch_pct'])
                        data.append((stretch_pct, F_N, np.full((3,2) ,np.nan), np.full(3, np.nan), np.full(2, np.nan), np.full(2, np.nan), mean_stretch, std_stretch))  
                    else:
                        data.append((stretch_pct, F_N, np.full((3,2) ,np.nan), np.full(3, np.nan), np.full(2, np.nan), np.full(2, np.nan)))  
                    
            
            
            except FileNotFoundError:
                print(f"<-- Missing file")
                
    print()
    
    # Organize data 
    data = np.array(data, dtype = 'object')
    rupture = np.array(rupture, dtype = 'object')
    
    # Order by stretch
    sort = np.argsort(data[:, 0])
    data = data[sort]
    rupture = rupture[sort]
    
    if data.shape[1] == 6:
        stretch_pct, F_N, Ff, Ff_std, contact_mean, contact_std = np.stack(data[:, 0]), np.stack(data[:, 1]), np.stack(data[:, 2]), np.stack(data[:, 3]), np.stack(data[:, 4]), np.stack(data[:, 5]) 
    elif data.shape[1] == 8:
        stretch_pct, F_N, Ff, Ff_std, contact_mean, contact_std, mean_stretch, std_stretch = np.stack(data[:, 0]), np.stack(data[:, 1]), np.stack(data[:, 2]), np.stack(data[:, 3]), np.stack(data[:, 4]), np.stack(data[:, 5]), np.stack(data[:, 6]), np.stack(data[:, 7]) 
    
    rup_stretch_pct, rup_F_N, rup, filenames = np.stack(rupture[:, 0]), np.stack(rupture[:, 1]), np.stack(rupture[:, 2]), np.stack(rupture[:, 3])
    
        
    
    # --- Rupture detection --- #
    if (rup > 0.5).any():
        # Print information
        detections = [["stretch %", "F_N", "Filenames"]]
        map = np.argwhere(rup == 1).ravel()
        for i in map:
            detections.append([rup_stretch_pct[i], rup_F_N[i], filenames[i].removeprefix(folder)])
        detections = np.array(detections)
        try:
            practical_rupture_stretch = np.min(detections[1:,0].astype('float'))
        except ValueError:
            # print(detections)
            exit()
        print(f"{len(detections)-1} Ruptures detected in \'{folder}\':")
        print(detections)                
    else:
        practical_rupture_stretch = None
        print("No rupture detected")
        
        
    output = {
        'stretch_pct': stretch_pct,
        'std_stretch': std_stretch,
        'F_N': F_N,
        'Ff': Ff,
        'Ff_std': Ff_std,
        'contact_mean': contact_mean,
        'contact_std': contact_std,
        'rup_stretch_pct': rup_stretch_pct,
        'rup_F_N': rup_F_N,
        'rup': rup,
        'filenames': filenames,
        'rupture_stretch': rupture_stretch,
        'practical_rupture_stretch': practical_rupture_stretch,
    }    
    
    if data.shape[1] == 8:
        output['mean_stretch'] = mean_stretch
        output['std_stretch'] = std_stretch
        
 
        
    return output
    
    


def manual_coupling(path, compare_path = None, save = False):
    """ Friction vs. normal force (F_N) for manual coupling between stretch and F_N """
    
    mean_window_pct = 0.5 # relative length of the mean window [% of total duration]
    std_window_pct = 0.35  # relative length of the std windoe [% of mean window]
    stretch_tension_file = 'stretch_tension.txt' 
    stretch_tension_file = 'stretch_tension_rupture_test.txt' 

    cmap = matplotlib.cm.viridis
    colorbar_scale = 'log'
    
    plotset_coupling = {'marker': 'v',
                        's': 40,
                        'edgecolors': "black"}
    plotset_compare =  {'marker': 'o',
                        's': 40,
                        'edgecolors': "black"}
   
    
    # Get load (tension) vs stretch
    stretch_tension = read_friction_file(os.path.join(path, stretch_tension_file))
    rupture_dict = read_info_file(os.path.join(path, 'rupture_test.txt'))
    stretch_test = stretch_tension['v_stretch_pct']
    load_test = metal_to_SI(stretch_tension['v_load'], 'F')*1e9
    tension_test = metal_to_SI(stretch_tension['v_tension'], 'F')*1e9
    
    
    
    fig, axes = plt.subplots(1, 3, num = unique_fignum(), figsize = (10,5), gridspec_kw ={'width_ratios': [1, 1, 0.05]})
    handles, labels = [], []
    
    
    # Get coupling data
    data = read_multi_coupling(path, mean_window_pct, std_window_pct)
    info = read_info_file(os.path.join(path, 'info_file.txt'))
    # stretch = data['stretch_pct']
    stretch = data['mean_stretch']
    stretch_initial = data['stretch_pct']
    std_stretch = data['std_stretch']
    non_rup = data['rup'] < 1
    
    
    # for i in range(len(stretch)):
    #         # print(f'{data["stretch_pct"][i]}, {data["mean_stretch"][i]}, {data["std_stretch"][i]}')
    #         print(f'{data["stretch_pct"][i]}, {data["F_N"][i]}, {data["filenames"][i]}')
            
            
    F_N = data['F_N'] # Full sheet
    Ff = data['Ff'][:, 0, 1]
    vmin = 0.1
    vmax = 10
    
    # Get min max for F_N    
    FN_min = np.min(F_N)
    FN_max = np.max(F_N)
    

    # Add compare data (without coupling)
    if compare_path is not None:
        data = read_multi_folder(compare_path, mean_window_pct, std_window_pct)    
        stretch_compare = data['stretch_pct']
        Ff_compare = data['Ff'][:, :, 0, 1]
        F_N_compare = data['F_N']
        
        a = FN_min
        b = np.min(F_N_compare)
        
        FN_min = min(FN_min, np.min(F_N_compare))
        FN_max = max(FN_max, np.max(F_N_compare))
        

    # Plot compare    
    if compare_path is not None:     
        for k in range(len(F_N_compare)):
                # color = get_color_value(F_N_compare[k], FN_min, FN_max, scale = colorbar_scale, cmap = cmap)
                color = get_color_value(F_N_compare[k], vmin, vmax, scale = colorbar_scale, cmap = cmap)
                axes[1].scatter(stretch_compare, Ff_compare[:,k], **plotset_compare, color = color, label = r'Const. $F_N$')
        h, l = axes[1].get_legend_handles_labels()
        handles.append(h[-1])
        labels.append(l[-1])
    
    # Plot coupling
    axes[0].scatter(F_N[non_rup], stretch[non_rup], marker = 'v', s = 40, edgecolors = "black",  color = color_cycle(1), label = "Mean stretch")
    axes[0].scatter(F_N[non_rup], stretch_initial[non_rup], marker = 'o', edgecolors = "black", color = color_cycle(0), alpha = 0.5, label = "Initial stretch")
    # axes[0].fill_between(F_N, stretch - 3*std_stretch, stretch + 3*std_stretch, color = color_cycle(2), alpha = 0.5, label = "$\pm 3\sigma$ ", zorder = -1)
   
    axes[1].scatter(stretch[non_rup], Ff[non_rup], **plotset_coupling, color = color_cycle(1), label = f"Coupled (R = {rupture_dict['R']:g})")
    add_xaxis(axes[1], x = stretch[non_rup], xnew = F_N[non_rup], xlabel = r'$F_N$ [nN] (Coupled)', decimals = 1, fontsize = 14)
    
 
    # for k in range(len(F_N)):
    #     color = get_color_value(F_N[k], FN_min, FN_max, scale = colorbar_scale, cmap = cmap)
    #     axes[1].scatter(stretch[k], Ff[k], **plotset_coupling, color = color, label = 'Coupled')
    h, l = axes[1].get_legend_handles_labels()
    handles.append(h[-1])
    labels.append(l[-1])
    

    # Colorbar
    if colorbar_scale == 'linear':
        # norm = matplotlib.colors.BoundaryNorm(np.linspace(FN_min, FN_max, 11), cmap.N)
        norm = matplotlib.colors.Normalize(vmin, vmax)
    elif colorbar_scale == 'log':
        # norm = matplotlib.colors.LogNorm(FN_min, FN_max)
        norm = matplotlib.colors.LogNorm(vmin, vmax)
    else:
        exit(f'scale = \'{colorbar_scale}\' is not defined.')
        
    
        
    axes[-1].grid(False)
    axes[-1].set_aspect(10)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axes[-1])
    cb.set_label(label = '$F_N$ [nN]', fontsize=14)


    axes[1].set_xlabel('Stretch', fontsize = 14)
    axes[1].set_ylabel(r'$\langle F_\parallel \rangle$ [nN]', fontsize = 14)
    
    # fig.legend(handles, labels, loc = 'upper left', fontsize = 13)
    axes[1].legend(handles, labels, loc = 'best', fontsize = 13)
    

    
    # plot load-stretch curve
    axes[0].plot(load_test, stretch_test, linewidth = 1, alpha = 1, label = 'Rupture test')
    add_xaxis(axes[0], x = load_test, xnew = load_test*rupture_dict['R'], xlabel = 'Tension [nN]', decimals = 1, fontsize = 14)
    axes[0].set_xlabel(r'$F_N$ [nN]', fontsize = 14)
    axes[0].set_ylabel('Stretch', fontsize = 14)
    axes[0].legend(fontsize = 13)


    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    
    
    if save is not False:
        # plt.savefig('../article/figures/negative_coefficient/manual_coupling.pdf', bbox_inches='tight')
        plt.savefig(f'../article/figures/negative_coefficient/{save}', bbox_inches='tight')
    

def manual_coupling_locked(save = False):
        
    path = '../Data/negative_coef/multi_coupling_popup'
    compare_path = '../Data/Baseline_fixmove/popup/multi_stretch'
    if save is not False:
        manual_coupling(path, compare_path, save = 'manual_coupling_pop1_7_5.pdf')
    else:
        manual_coupling(path, compare_path, save)
        
    path = '../Data/negative_coef/multi_coupling_honeycomb'
    compare_path = '../Data/Baseline_fixmove/honeycomb/multi_stretch'
    if save is not False:
        manual_coupling(path, compare_path, save = 'manual_coupling_hon3215.pdf')
    else:
        manual_coupling(path, compare_path, save)



def manual_coupling_free(save = False):        
    path = '../Data/negative_coef/multi_coupling_free_popup'
    compare_path = '../Data/Baseline_fixmove/popup/multi_stretch'
    if save is not False:
        save = 'manual_coupling_free_pop1_7_5.pdf'
    manual_coupling(path, compare_path, save)
        
    
    path = '../Data/negative_coef/multi_coupling_free_honeycomb'
    compare_path = '../Data/Baseline_fixmove/honeycomb/multi_stretch'
    if save is not False:
        save = 'manual_coupling_free_hon3215.pdf'
    manual_coupling(path, compare_path, save)
        
    
    
 
    
    # path = '../Data/negative_coef/multi_coupling_honeycomb'
    # compare_path = '../Data/Baseline_fixmove/honeycomb/multi_stretch'
    # manual_coupling(path, compare_path, save = 'manual_coupling_hon3215.pdf')
    
    

if __name__ == '__main__':
    # manual_coupling_locked(save = False)
    manual_coupling_free(save = True)
    
    
    plt.show()