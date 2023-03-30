import sys
sys.path.append('../') # parent folder: MastersThesis

if 'MastersThesis' in sys.path[0]: # Local 
    from ML.dataloaders import *
    from analysis.analysis_utils import *
else: # Cluster
    from dataloaders import *
    from analysis_utils import *
    

from plot_set import *
import ast

import seaborn as sns


class Data_fetch():
    """ Fetch data for analysing (not ML) """
    
    # Partial inheritance from KirigamiDataset 
    collect_dirs = KirigamiDataset.collect_dirs
    __len__ = KirigamiDataset.__len__
    
    def __init__(self, data_root):
        self.data_root = data_root
        self.data_dir = []
        self.sample = None

        # --- Collect data dirs --- #
        indexes = [] # Store index from data names
        if isinstance(data_root, str): # Single path [string]
            indexes = self.collect_dirs(data_root, indexes)
            
        elif  hasattr(data_root, '__len__'): # Multiple paths [list of strings]
            for path in data_root:
                indexes = self.collect_dirs(path, indexes)
                
        else:
            print(f"Data root: {data_root}, not understood")
            exit()
        
        
    def get_data(self, target, idx, exclude_rupture = True):
        # stretch, F_N, scan_angle, dt, T
        # drag_speed, drag_length, K, stretch_speed_pct	
        # relax_time, pause_time1, pause_time2, rupture_stretch, is_ruptured	
        # Ff_max, Ff_mean, Ff_mean_std, contact,  contact_std,       
        
        self.data = {}
        if exclude_rupture:
            self.data['is_ruptured'] = []
        for key in target:
            self.data[key] = []
        
    
        for i in idx:
            with open(os.path.join(self.data_dir[i], 'val.csv'), newline='') as csvfile:
                reader = csv.reader(csvfile)
                for key, val in reader:
                    if key in self.data:
                        self.data[key].append(val)
        
        for key in self.data:
            self.data[key] = np.array(self.data[key], dtype = float)
        
        
        # Exclude rupture events due to nan values for some variables
        if exclude_rupture:
            rupture_mask = self.data['is_ruptured'] > 0.5
            for key in self.data:
                self.data[key] = self.data[key][~rupture_mask]     
            self.data.pop('is_ruptured')
        
        return self.data
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return np.array([idx])
        else:
            return np.arange(len(self))[idx]
        
    

    def corrcoef_matrix(self, label_map = None, save = False):
        """ Pearson product-moment correlation coefficients. """
        
        
        var = np.array([self.data[key] for key in self.data])
        names = [key for key in self.data]
        
        if label_map is not None:
            label_map_keys = np.array([key for key in label_map])
            label_map_sort = []
            for i, key in enumerate(names):
                if key in label_map:
                    label_map_sort.append(np.argwhere(label_map_keys == names[i]).ravel()[0])
                    names[i] = label_map[key]
                else:
                    label_map_sort.append(len(names)-1)
        
        
        # Sort 
        sort = np.argsort(label_map_sort)
        old_names = names.copy()
        for i, j in enumerate(sort): # Sort list
            names[i] = old_names[j]
        var = var[sort] # Sort array
        
        assert var.shape[1] > 1, f"The number of datapoints per variable = {var.shape[1]} must be more than one"
        
        
        mat = np.corrcoef(var)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(mat, xticklabels=names, yticklabels = names, cbar_kws={'label': 'Correlation coefficient'}, annot=True, vmin=-1, vmax=1, cmap = 'coolwarm', ax=ax)
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        if save is not None:
            plt.savefig(f'../article/figures/ML/{save}', bbox_inches='tight')
            
            

        plt.show()
    


def plot_corrcoef(save = False):
    root = '../Data/ML_data/'
    data_root = [root+'baseline', root+'popup', root+'honeycomb', root+'RW']
    obj = Data_fetch(data_root)
    
    obj.get_data(['stretch', 'rupture_stretch', 'F_N', 'porosity', 'Ff_mean',  'Ff_max', 'contact', 'Ff_mean_std', 'contact_std'], obj[:])
    obj.data['rel. stretch'] =  obj.data['stretch']/obj.data['rupture_stretch']
    
    if save:
        savename = 'corrcoef_matrix.pdf'
    else:
        savename = False
    
    mat = obj.corrcoef_matrix({'porosity': 'porosity',
                               'stretch': 'stretch',  
                               'rel. stretch': 'rel. stretch',
                               'rupture_stretch': 'rup. stretch',
                               'F_N': '$F_N$', 
                               'Ff_mean': r'$\langle F_\parallel \rangle$', 
                               'Ff_max': 'max Ff',
                               'contact': 'contact',
                               'Ff_mean_std': r'std $\langle F_\parallel \rangle$',
                               'contact_std': 'std contact'
                               }, save = savename)
    
    
def plot_corr_scatter(save = False):
    root = '../Data/ML_data/'
    data_root = [root+'popup', root+'honeycomb', root+'RW']
    reg_name = ['Tetrahedron    ', 'Honeycomb    ', 'Random Walk']
    
    
    size = 5
    plot_settings = [{'color': color_cycle(1), 'marker': '^', 's': size, 'alpha': 0.5},
                     {'color': color_cycle(3), 'marker': 'D', 's': size, 'alpha': 0.5},
                     {'color': color_cycle(4), 'marker': 'o', 's': size, 'zorder': -1, 'alpha': 0.5}]
    

    
    
    # x, y, x-label, y-label, savename
    plots = [
    ['stretch'          , 'Ff_mean', 'Stretch'              , r'$\langle F_\parallel \rangle$', 'corr_stretch_Ff.pdf'       ],
    ['rel. stretch'     , 'Ff_mean', 'rel. stretch'         , r'$\langle F_\parallel \rangle$', None                        ],
    ['porosity'         , 'Ff_mean', 'Porosity'             , r'$\langle F_\parallel \rangle$', 'corr_porosity_Ff.pdf'      ],
    ['contact'          , 'Ff_mean', 'Contact'              , r'$\langle F_\parallel \rangle$', 'corr_contact_Ff.pdf'       ],
    ['stretch'          , 'contact', 'Stretch'              , 'Contact'                       , 'corr_stretch_contact.pdf'  ],
    ['porosity'          ,'contact', 'Porosity'             , 'Contact'                       , 'corr_porosity_contact.pdf'  ],
    ]
    
    
    
    figs_axes = np.array([plt.subplots(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k') for _ in range(len(plots))])
    figs = figs_axes[:, 0]; axes = figs_axes[:, 1]
    
    
    for reg, reg_data in enumerate(data_root):
        obj = Data_fetch(reg_data)
        obj.get_data(['stretch', 'rupture_stretch', 'F_N', 'porosity', 'Ff_mean', 'contact', 'Ff_mean_std', 'contact_std'], obj[:])
        obj.data['rel. stretch'] =  obj.data['stretch']/obj.data['rupture_stretch']


        for ax_num in range(len(plots)):
            data = np.stack([obj.data[plots[ax_num][0]], obj.data[plots[ax_num][1]]])
            corrcoef = np.corrcoef(data)[0, 1]
            axes[ax_num].scatter(data[0], data[1], **plot_settings[reg], label = f'{reg_name[reg]} (corr = {corrcoef:0.3f})')
    
    
    
    # labels 
    for ax_num in range(len(plots)):
        axes[ax_num].set_xlabel(plots[ax_num][2], fontsize=14)
        axes[ax_num].set_ylabel(plots[ax_num][3], fontsize=14)
    
    for i, fig in enumerate(figs):
        fig.legend(fontsize = 13)
        fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        if save:
            savename = plots[i][-1]
            if savename is not None:
                fig.savefig(f'../article/figures/ML/{savename}', bbox_inches='tight')
        
        
def read_best_scores(path):
    infile = open(path, 'r')
    info = {}
    data = {}
    last_key = ''
    for line in infile:
        if '---' in line:
            header = line.strip('#-\n ')
            header = header.replace(' ', '_')
            info[header] = {}
            last_key = header
            continue
      
        if line[0] == '#':
            if len(last_key) > 0:
                key, val = line.strip('#\n ').split(' = ')
                
                # Convert to right data type
                if '[' in val or '(' in val:
                    if not ',' in val:
                        val = val.replace(' ', ', ')
                    try:
                        val = ast.literal_eval(val)
                    except ValueError:
                        pass # just string then
                else:
                    if val == 'True' or val == 'False':
                        val = bool(val)
                    else:
                        try:
                            val = int(val)
                        except ValueError:
                            pass # just string then
                
                info[last_key][key] = val 
                # model_settings[key] = val
        else:
            key, val = line.strip('\n').split(' ')
            data[key] = float(val)
            
    return info, data    
        
def model_performance(path, save = False):
    # Get folders
    folders = os.listdir(path)
    if '.DS_Store' in folders:
        folders.remove('.DS_Store')
    
    # Initialize arrays
    name = ['' for i in range(len(folders))]
    num_params = np.zeros(len(folders))
    epoch = np.zeros(len(folders))
    R2 = np.zeros((len(folders), 6))
    acc = np.zeros(len(folders))
    
    # Go through data
    for i, folder in enumerate(folders):
        # Read file
        file = os.path.join(path, folder, 'best_scores.txt')
        info, data = read_best_scores(file)
        
        # Add to arrays
        name[i] = info['Model_settings']['name']
        num_params[i] = info['Model_info']['num_params']
        epoch[i] = data['epoch']
        R2[i] = [data[f'R2_{i}'] for i in range(len(R2[i]))]
        acc[i] = data['acc']

    sort = np.argsort(num_params)
    
    num_params = num_params[sort]
    epoch = epoch[sort]
    R2 = R2[sort]
    acc = acc[sort]
    
    
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    plt.plot(R2[:, 0], '-o', label = r'$R_2$ $\langle F_\parallel \rangle$')
    plt.plot(R2[:, 1], '-o', label = r'$R_2$ $ \max F_\parallel$')
    plt.plot(R2[:, 2], '-o', label = r'$R_2$ Contact')
    plt.plot(R2[:, 3], '-o', label = r'$R_2$ Porosity')
    plt.plot(R2[:, 4], '-o', label = r'$R_2$ Rupture stretch')
    plt.plot(acc, '-o', label = 'Accuracy rupture')
    plt.xlabel(r'Model number', fontsize=14)
    plt.ylabel('Validation performance', fontsize=14)
    plt.legend(fontsize = 13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    # plt.savefig('../article/figures/figure.pdf', bbox_inches='tight')
    
    for i in range(len(name)):
        print(f'{i:2d} | name = {name[i]:30s}, #params = {int(num_params[i]):8d}, best epoch = {int(epoch[i]):4d}, R2 = {R2[i]}')



def get_rupture_count():
    root = '../Data/ML_data/'
    data_root = [root+'baseline', root+'popup', root+'honeycomb', root+'RW']
    tot_rup = 0
    tot_tot = 0
    for path in data_root:
        obj = Data_fetch(path)
        data = obj.get_data(['is_ruptured'], obj[:], exclude_rupture = False)
        is_rup = data['is_ruptured']
        rup = int(np.sum(is_rup))
        total = np.sum(is_rup > -1)
        pct = 100*rup/total
        print(f'path = {path}, rup = {rup}/{total} ({pct:0.2f}%)')
        tot_rup += rup
        tot_tot += total

    tot_pct = 100*tot_rup/tot_tot
    print(f'Total, rup = {tot_rup}/{tot_tot} ({tot_pct:0.2f}%)')
    
        
        
if __name__ == '__main__':
    # plot_corrcoef(save = False)
    # plot_corr_scatter(save = False)
    
    # model_performance('training_1')
    # model_performance('training_3')

    # get_rupture_count()
    
    # plt.show()
    pass
    
    # data_root = ['../Data/ML_data/honeycomb', '../Data/ML_data/popup'] 
    # obj = Data_fetch(data_root)
    # data = obj.get_data(['is_ruptured'], obj[:], exclude_rupture = False)
    # part = np.sum(data['is_ruptured']) / len(data['is_ruptured'])
    # print(part)
    
    
    