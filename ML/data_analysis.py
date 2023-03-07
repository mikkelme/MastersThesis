import sys
sys.path.append('../') # parent folder: MastersThesis
from plot_set import *
from analysis.analysis_utils import *

from dataloaders import *
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
        
        
        var = np.array([self.data[keys] for keys in self.data])
        names = [keys for keys in self.data]
        
        if label_map is not None:
            for i, key in enumerate(names):
                if key in label_map:
                    names[i] = label_map[key]
        
        
        assert var.shape[1] > 1, f"The number of datapoints per variable = {var.shape[1]} must be more than one"
        
        
        mat = np.corrcoef(var)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(mat, xticklabels=names, yticklabels = names, cbar_kws={'label': 'Correlation coefficients'}, annot=True, vmin=-1, vmax=1, cmap = 'coolwarm', ax=ax)
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        if save is not None:
            plt.savefig(f'../article/figures/ML/{save}', bbox_inches='tight')
            
            

        plt.show()
    




def plot_corrcoef(save = False):
    data_root = ['../Data/ML_data/honeycomb', '../Data/ML_data/popup'] 
    obj = Data_fetch(data_root)
    
    obj.get_data(['stretch', 'rupture_stretch', 'F_N', 'porosity', 'Ff_mean',  'Ff_max', 'contact', 'Ff_mean_std', 'contact_std'], obj[:])
    obj.data['rel. stretch'] =  obj.data['stretch']/obj.data['rupture_stretch']
    
    if save:
        savename = 'corrcoef_matrix.pdf'
    else:
        savename = False
    
    mat = obj.corrcoef_matrix({'stretch': 'Stretch', 
                               'rupture_stretch': 'rup. stretch',
                               'F_N': '$F_N$', 
                               'porosity': 'Porosity', 
                               'Ff_mean': r'$\langle F_\parallel \rangle$', 
                               'Ff_max': 'max Ff',
                               'contact': 'Contact',
                               'Ff_mean_std': r'std $\langle F_\parallel \rangle$',
                               'contact_std': 'std contact'
                               }, save = savename)
    
    
    
def plot_corr_scatter(save = False):
    data_root = ['../Data/ML_data/honeycomb', '../Data/ML_data/popup'] 
    
    size = 15
    plot_settings = [{'color': color_cycle(1), 'marker': '^', 's': size},
                     {'color': color_cycle(3), 'marker': 'D', 's': size}]
    
    reg_name = ['popup', 'honeycomb']

    
    # x, y, x-label, y-label, savename
    plots = [
    ['stretch'          , 'Ff_mean', 'Stretch'              , r'$\langle F_\parallel \rangle$', 'corr_stretch_Ff.pdf'       ],
    ['rel. stretch'     , 'Ff_mean', 'rel. stretch'         , r'$\langle F_\parallel \rangle$', None                        ],
    ['porosity'         , 'Ff_mean', 'Porosity'             , r'$\langle F_\parallel \rangle$', 'corr_porosity_Ff.pdf'      ],
    ['contact'          , 'Ff_mean', 'Contact'              , r'$\langle F_\parallel \rangle$', 'corr_contact_Ff.pdf'       ],
    ['stretch'          , 'contact', 'Stretch'              , 'Contact'                       , 'corr_stretch_contact.pdf'  ],
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
        
        
        
    
        
# plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
# plt.xlabel(r'$x$', fontsize=14)
# plt.ylabel(r'$y$', fontsize=14)

        

        

if __name__ == '__main__':
    
    # plot_corrcoef(save = False)
    # plot_corr_scatter(save = True)
    plt.show()
    pass
    
    # data_root = ['../Data/ML_data/honeycomb', '../Data/ML_data/popup'] 
    # obj = Data_fetch(data_root)
    # data = obj.get_data(['is_ruptured'], obj[:], exclude_rupture = False)
    # part = np.sum(data['is_ruptured']) / len(data['is_ruptured'])
    # print(part)
    
    
    