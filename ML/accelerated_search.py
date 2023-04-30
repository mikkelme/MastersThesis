import sys
sys.path.append('../') # parent folder: MastersThesis

from ML.use_network import *
from config_builder.build_config import *
from graphene_sheet.build_utils import *
from ase.visualize.plot import plot_atoms

from time import perf_counter

# from graphene_sheet.RN_walks import *



class Genetic_algorithm: # Genetic algorithm 
    def __init__(self, model_weights, model_info, N = 100, image_shape = (62, 106), expand = None, repair = False):
        if model_weights is None:
            return # when used to get repair function
        
        
        # Settings        
        self.N = N
        self.image_shape = image_shape
        self.expand = expand
        self.repair = repair
        
        # Initialize arrays
        self.A = np.zeros((N, *image_shape), dtype = int)   # Population
        self.P = np.zeros((*self.image_shape, 2,2))         # Site transistion probabilities 
        self.n0 = np.zeros(self.image_shape)                # Site distribution states
        self.n0_target = np.zeros(self.image_shape)         # Site target distribution states
        self.scores = np.zeros(self.N)                      # Population scores
        
        # Set generation to 0
        self.gen = 0
        
        # Initialize evaluater
        self.EV = Evaluater(model_weights, model_info)
    
        if expand is not None:
            assert self.expand[0] > self.image_shape[0]
            assert self.expand[1] > self.image_shape[1]
            
            dx = self.expand[0]  - self.image_shape[0]
            dy = self.expand[1]  - self.image_shape[1]
            self.offset = (dx//2, dy//2)
            self.x_center = [self.offset[0], self.offset[0] + self.image_shape[0]]
            self.y_center = [self.offset[1], self.offset[1] + self.image_shape[1]]
            
            self.A_ex =  np.zeros((N, *expand), dtype = int)   # Expanded Population 
            
            
    
        #########################
        # self.N_mark = self.N//2
        self.N_mark = self.N//10
        # self.N_mark = self.N//10

        
        
    def init_population(self, configs):
        assert isinstance(configs, list), f"input_population input must be a list (containing str, array and/or float) not type: {type(configs)}."
        for i in range(self.N):
            conf = configs[i%len(configs)]
            if isinstance(conf, str): # Path to array
                self.A[i] = np.load(conf).astype(np.float32)
            elif isinstance(conf, np.ndarray): # Array
                self.A[i] = conf.copy()
            elif isinstance(conf, float): # Float defining porosity probability 
                ones = np.random.rand(*self.image_shape) > conf
                self.A[i][ones] = 1
                
            if self.repair:
                self.A[i] = self.repair_sheet(self.A[i])
        
        if self.expand:
            self.expand_population()
            
    
    def expand_population(self):
        for i in range(self.N):
            for x in range(-self.x_center[0], self.expand[0]-self.x_center[0]):
                for y in range(-self.y_center[0], self.expand[1]-self.y_center[0]):
                    Ax, Ay = (x + self.image_shape[0])%self.image_shape[0],  (y + self.image_shape[1])%self.image_shape[1]
                    self.A_ex[i, x + self.x_center[0], y + self.y_center[0]] = self.A[i, Ax, Ay]

    def set_fitness_func(self, func):
        self.fitness = func
        
        
    def max_drop(self, conf): 
        self.EV.set_config(conf)
        metrics = self.EV.evaluate_properties(self.stretch, self.F_N)
        score = metrics['Ff_max_drop'][-1] 
        return score
    
    def max_fric(self, conf): 
        self.EV.set_config(conf)
        metrics = self.EV.evaluate_properties(self.stretch, self.F_N)
        score = metrics['Ff_max'][-1] 
        return score
    

    
    def evaluate_fitness(self):
        for i in range(self.N):
            if self.expand is not None:
                self.scores[i] = self.fitness(self.A_ex[i])
            else:
                self.scores[i] = self.fitness(self.A[i])
        
        # Get rank
        self.rank = np.argsort(self.scores)[::-1] # In descending order
        
        # Sort scores and population 
        self.scores = self.scores[self.rank]
        self.A = self.A[self.rank]
        
        
        if self.expand is not None:
            self.A_ex = self.A_ex[self.rank]
        
        
        # Store min, mean and max score
        self.min_score  = self.scores[-1]
        self.mean_score = np.mean(self.scores)
        self.max_score  = self.scores[0]
           
      
    def update_state_distribution(self):
        C1 = np.mean(self.A, axis = 0)
        C0 = 1 - C1
        self.n0 = C0
        
    def update_state_distribution_target(self):
        C0_target = np.max(np.multiply(self.W[:, np.newaxis, np.newaxis], -self.A+1), axis = 0)
        C1_target = np.max(np.multiply(self.W[:, np.newaxis, np.newaxis], self.A), axis = 0)
        # C0_target = np.sum(np.multiply(self.W[:, np.newaxis, np.newaxis], -self.A+1), axis = 0)
        # C1_target = np.sum(np.multiply(self.W[:, np.newaxis, np.newaxis], self.A), axis = 0)
        
        # Normalize
        self.n0_target = C0_target /(C0_target + C1_target)
    
    def update_gene_transistion_probabilities(self):
        
        self.update_state_distribution()

        # --- Set P00 --- #
        if self.gen == 0:
            n0 = self.n0
            self.P[:, :, 0, 0] = 0.5
            self.update_state_distribution_target()
        else:
            # n0 = self.n0_target # Old target
            n0 = (self.n0_target+self.n0)/2 # mix
            self.update_state_distribution_target()
            self.P[:, :, 0, 0] = n0
        
        
        # n0 = self.n0
        # --- Calculate P10 --- #
        n1 = 1 - n0
        nonzero_n1 = n1 > 0
        self.P[nonzero_n1, 1, 0] = (self.n0_target[nonzero_n1] - self.P[nonzero_n1, 0, 0] * n0[nonzero_n1])/n1[nonzero_n1]
        self.P[~nonzero_n1, 1, 0] = 1
        

        
        # --- Calculate P01 --- #
        self.P[:, :, 0, 1] = 1 - self.P[:, :, 0, 0]
        
        # --- Clip --- #
        # Clip the result at probability range [0, 1]
        self.P[self.P[:, :, 1, 0] < 0, 1, 0] = 0
        self.P[self.P[:, :, 1, 0] > 1, 1, 0] = 1
        
    
        
    def ranking_func(self):
        
        # Linear
        i = np.arange(self.N).astype('float')
        r = np.where(i <= self.N_mark, i, 1)
        
        # Linear interpolaiton of scores
        # i = np.arange(self.N)
        # best_score = self.scores[0]
        # mark_score = self.scores[self.N_mark+1]
        # slope = 1/(mark_score-best_score)
        # r = np.where(i <= self.N_mark, np.abs((self.scores[i]-best_score)*slope) , 1)
        
        # Quadratic    
        # r = np.where(i <= self.N_mark, i**2, 1)
        
        # Log
        # r = np.where(i <= self.N_mark,np.log((i+1)), 1)
        
        # Normalize 
        r[:self.N_mark+1] /= r[self.N_mark]
        return r
        
    def mutate(self):
        r = self.ranking_func()
        a = r
        self.W = 1 - a
       
       
        # mutate_individual = np.random.rand(self.N) < a
        
        self.update_gene_transistion_probabilities()
        
        # Clip bottom prob. increasingly towards end to avoid locking of prob.
        i = np.arange(self.N).astype('float')
        max_bound = 0.05
        slope = max_bound/(self.N-self.N_mark-1)
        clip = np.where(i >= self.N_mark, (i - self.N_mark)*slope, 0)
        

        for i in range(self.N): 
            RN = np.random.rand(*self.image_shape)
            zeros = self.A[i] < 0.5
            flip0 = np.logical_and(RN < np.maximum(self.P[:, :, 0, 1]*a[i], clip[i]), zeros)
            flip1 = np.logical_and(RN < np.maximum(self.P[:, :, 1, 0]*a[i], clip[i]), ~zeros)
            self.A[i][flip0] = 1
            self.A[i][flip1] = 0
            
            if self.repair:
                self.A[i] = self.repair_sheet(self.A[i])

            if self.expand is not None:
                self.expand_population()    
                    
        
        
    def evolve(self):
        """ Evolve by one generation """
        
        self.mutate()
        self.gen += 1
        self.evaluate_fitness()
        
        # No crossover for now
        # Mutate
        
    
    def evolution(self, num_generations = 1000):
        
        timer_start = perf_counter() 

        
        self.evaluate_fitness()
        for generation in range(num_generations):
            try:
                best_porosity = 1-np.mean(self.A[0])
                print(f'Gen = {self.gen} | Min score = {self.min_score:g}, Mean score = {self.mean_score:g}, Max score = {self.max_score:g}, mean P01 = {np.mean(self.P[:, :, 0, 1]):g}, mean P10 = {np.mean(self.P[:, :, 1, 0]):g}, porosity = {best_porosity:g}')
                
                # if self.gen % 10 == 0:
                #     fig = self.show_status()
                #     fig.savefig(f'AS/gen{self.gen}.pdf', bbox_inches='tight')
                #     # plt.show()
                #     fig.clf()
                    
                    # self.show_sheet()
                # if self.gen % 100 == 1:
                #     self.show_status()
                #     plt.show()
                    
                    
                self.evolve()
            except KeyboardInterrupt: 
                break
        
        timer_stop = perf_counter()
        elapsed_time = timer_stop - timer_start
        h = int(elapsed_time // 3600)
        m = int((elapsed_time % 3600) // 60)
        s = int(elapsed_time % 60)
        self.s_timing = f'{h:02d}:{m:02d}:{s:02d}'
        print(f'Elapsed time: {self.s_timing}')
        
    def show_sheet(self, conf):
        builder = config_builder(conf)
        builder.view()
        
    def show_status(self):
        fig, axes = plt.subplots(3, 2, num = unique_fignum(), figsize = (12, 8))

        if self.gen == 0:
            fig.suptitle(f'Gen = {self.gen}')
        else:
            fig.suptitle(f'Gen = {self.gen} | Min score = {self.min_score:g}, Mean score = {self.mean_score:g}, Max score = {self.max_score:g}, mean P01 = {np.mean(self.P[:, :, 0, 1]):g}, mean P10 = {np.mean(self.P[:, :, 1, 0]):g}')


        if self.image_shape == (62, 106) or self.expand == (62, 106):   
            if self.expand is not None:
                A = self.A_ex[0]
            else:
                A = self.A[0]
            builder = config_builder(A)
            GA.EV.set_config(A)
            plot_atoms(builder.sheet, axes[0, 0], radii = 0.8, show_unit_cell = 0, scale = 1, offset = (0,0))
            GA.EV.stretch_profile(GA.stretch, GA.F_N, axes[0, 1])
            axes[0, 1].set_ylim(top=3.0)
        else:
            if self.expand is not None:
                axes[0,1].imshow(self.A_ex[0], vmin = 0, vmax = 1, origin = 'lower')
            else:
                axes[0,1].imshow(self.A[0], vmin = 0, vmax = 1, origin = 'lower')
        

        # self.show_sheet()
        axes[1,0].imshow(1 - self.n0, vmin = 0, vmax = 1, origin = 'lower')
        axes[1,0].set_title(r'$1 - n_0$')
        axes[1,1].imshow(1- self.n0_target, vmin = 0, vmax = 1, origin = 'lower')
        axes[1,1].set_title(r'$1 -n_{0, t+1}$')
        
        axes[2,0].imshow(self.P[:, :, 1, 0], vmin = 0, vmax = 1, origin = 'lower')
        axes[2,0].set_title(r'$P_{10}$')
        
        axes[2,1].imshow(self.P[:, :, 0, 1], vmin = 0, vmax = 1, origin = 'lower')
        axes[2,1].set_title(r'$P_{01}$')
        fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        
        
        return fig
       
    def repair_sheet(self, conf, max_walk_dis = None):
        """ Repair sheet configuration by reducing it to a single cluster that 
            spans top to bottom in y-direction.

        Args:
            conf (2D array): Cut configuration
            max_walk_dis (None or int, optional): Maximum number of steps for walkers to take. Defaults to None meaning that the current cluster size is used.


        Algorithm summary:
                1.  Update label in descending order by cluster size.
                2.  Start walker from smallest cluster (last element of list).
                3.  Walk until another cluster is found or the step size 
                    exceeds the cluster size itself. 
                4.  If (another cluster was found)  --> connect thoose
                    Else:
                        If (cluster is the last cluster attached)   --> Extend walking
                        Else                                        --> Remove cluster
                
                5.  If number of clusters is 1 --> Terminate
                    Else:                      --> Go to step 1.
                    
        """
       
        # --- Check that top and bottom row is present --- #
        # If missing fill row completely 
        num_top_atoms = np.sum(conf[:, -1])
        num_bottom_atoms = np.sum(conf[:, 0])
        if num_top_atoms == 0:
            conf[:, -1] = 1
        if num_bottom_atoms == 0:
            conf[:, 0] = 1
            
        # --- Get clusters --- #
        self.labels, self.cluster_sizes = self.get_clusters(conf)
        self.num_clusters = len(self.cluster_sizes)
        self.min_dis = 1 # For DFS
        
        
        while self.num_clusters > 1:
            # Order clusters by size and pick on from minimum
            self.reorder()
            label = self.labels[-1]
            
            # Get current cluster size
            size = self.cluster_sizes[-1] 
            if max_walk_dis is not None: # Adjust for cases of extended walk
                size = max_walk_dis
            
            # Get copy of initial state
            self.visit_old = self.visit.copy() 
            
            # Reset parameters
            num_atoms_added = 0
            max_path_len = 0
            num_clusters = self.num_clusters
            
            # Set new starting point from edge
            edge = self.get_edge(label) 
            path = [[e] for e in edge]
            best_label = [-1 for e in edge]
            
            # --- Walk from smallest cluster --- #
            while num_clusters > 1 and max_path_len <= size:         
                # Remove possible duplicates on last site in path
                last_elements = [l[-1] for l in path]
                k = 0; k_end = len(path)
                while k < k_end:
                    match = np.all(path[k][-1] == last_elements, axis = 1)
                    match[k] = False
                    del_idx = np.argwhere(match).ravel()
                    for pop_count, d in enumerate(del_idx):
                        del path[d-pop_count]
                        del last_elements[d-pop_count]
                        del best_label[d-pop_count]
                        k_end -= 1
                    k += 1
                
                # Update max path length 
                max_path_len = len(path[0]) + num_atoms_added


                # If any positive labels is present (HIT) 
                if np.max(best_label) > 0:
                    hit = np.argmax(best_label)
                    p = np.array(path[hit])
                    l = best_label[hit]
                    num_atoms_added += len(p) - 1 
                    max_path_len +=  len(p) - 1
                    from_label = np.max((l, label))
                    to_label = np.min((l, label))
                    assert from_label > to_label, "Pretty sure this should always be the case now..."
                    self.visit[p[:,0], p[:,1]] = to_label
                    self.visit[self.visit == from_label] = to_label
                    
                    # Merge clusters sizes
                    # Note: We only merge original sizes not the one from added atoms
                    from_idx = from_label - 1
                    to_idx = to_label - 1                    
                    self.cluster_sizes[to_idx] += self.cluster_sizes[from_idx] 
                    
                    # Remove from cluster_sizes ans labels
                    self.cluster_sizes.pop(from_idx) 
                    self.labels.pop(from_idx)
                    num_clusters -= 1
                    # print(f'label = {label}, build: {p}, num_clusters: {self.num_clusters}->{num_clusters}')
                    
                    break # while loop for current label
                        
                # If max walking length is reached check of clusters is last spanning option
                if max_path_len > size: 
                    del_map = self.visit == label
                    
                    # Potential top and bottom atoms left when removing cluster
                    top_atoms_left = np.sum(self.visit[:, -1] > 0) - np.sum(del_map[:, -1])
                    bottom_atoms_left = np.sum(self.visit[:, 0] > 0) - np.sum(del_map[:, 0])
                    
                    # If this is killing spanning properties exten walking (by increasing cluster size)
                    if top_atoms_left == 0 or bottom_atoms_left == 0: # Killing spanning properties
                        # print(f"label = {label}, go nuts")
                        size += 1 # Unlimited walking allowed to find connection
            

                # Add new site to paths
                for i in range(len(path)):
                    p = path[i]
                    
                    # Get current (end of path) position and label
                    current_pos = p[-1]
                    current_site_label = self.visit[current_pos[0], current_pos[1]]
                    
                    
                    # Walk from end of path
                    walk = np.array(self.walk_dis([current_pos], label = label))
                    
                    # Remove elements already in the preceding part of the path    
                    not_in_path = ~np.any(np.all(walk == np.array(p)[:, np.newaxis], axis = -1), axis = 0)
                    walk = walk[not_in_path]
                
                    # Get labels on destination sites
                    site_labels = self.visit[walk[:, 0], walk[:, 1]]
                    
                    # Store all path combinations
                    if len(walk) > 0:
                        for w in range(1, len(walk)):
                            path.append(path[i] + [walk[w]])
                            best_label.append(site_labels[w])
                        path[i].append(walk[0])
                        best_label[i] = site_labels[0]
        
        
            ### Coming out of while loop ###        
         
            if num_clusters == self.num_clusters: # Did not manage to reduce number of clusters
                # Remove cluster
                # print(f'label {label}, removing cluster')
                self.visit = self.visit_old # Reset visit array
                self.visit[self.visit == label] = -1 # Delete label cluster
                num_clusters -= 1
            
                label_idx = np.argmin(np.abs(label - np.array(self.labels)))
                self.labels.pop(label_idx) # TODO: Thos can most likely be removed ...
                self.cluster_sizes.pop(label_idx) 
                

            # Update number of clusters
            self.num_clusters = num_clusters
            
          
        # print(f'Repair completed') 
        
        # Update configuration
        conf[:] = 1
        conf[self.visit < 0] = 0
        return conf
    
    def walk_dis(self, input, label, dis = 0, pre = []):
        """ Recursive function to walk to all sites
            within a distance of min_dis jumps """
        if self.min_dis == 0:
            return input
        
        
        for i, elem in enumerate(input):
            if isinstance(elem, (np.ndarray, np.generic)):
                input[i] = elem.tolist()

        neigh = []
        for pos in input:
            suggest, _ = connected_neigh_atom(pos)
            for s in suggest:
                if len(pre) == 0:
                    s_in_pre = False
                else:
                    s_in_pre = np.any(np.all(s == pre, axis = -1))
                
                if len(neigh) == 0:
                    s_in_neigh = False
                else:
                    s_in_neigh = np.any(np.all(s == neigh, axis = -1))
            
                on_sheet =  np.all(np.logical_and(s < np.shape(self.visit), s >= (0,0)))
                if on_sheet:
                    site_label = self.visit[s[0], s[1]]
                else:
                    continue
                
                if not s_in_pre and not s_in_neigh and site_label != label:
                        neigh.append(s)
            
        dis += 1
        if dis >= self.min_dis:
            return input + neigh
        else:
            pre = np.array(input)
            return  np.concatenate((pre, self.walk_dis(neigh, label, dis, pre)))
    
    def reorder(self):
        """ Rearange labels to match cluster size
            Descending order: Biggest cluster first """
        self.cluster_sizes = np.array(self.cluster_sizes) # TODO: Do we want to avoid transformation between array and list here?
        zero_map = np.argwhere(self.visit < 0)
        size_sort = np.argsort(self.cluster_sizes)[::-1]
        
        for i, from_label in enumerate(size_sort+1):
            to_label = i+1
            self.visit[self.visit == from_label] = -to_label
        self.visit = np.abs(self.visit)
        self.visit[zero_map[:, 0], zero_map[:, 1]] = -1
    
        self.labels = [c for c in range(1, 1+len(self.cluster_sizes))]
        self.cluster_sizes = list(self.cluster_sizes[size_sort])
        
    def get_edge(self, label):
        """ Find edge of specific cluster """
        out = np.argwhere(self.visit == label)
        edge = []
        for pos in out:
            neigh, _ = connected_neigh_atom(pos)
            on_sheet = np.all(np.logical_and(neigh < np.shape(self.visit), neigh >= (0,0)), axis = 1)
            neigh = neigh[on_sheet]
            trial = self.visit[neigh[:, 0], neigh[:, 1]]
            # out = np.argwhere(trial == -1).ravel()
            out = np.argwhere(trial != label).ravel()
            for hit in neigh[out]:
                if len(edge) == 0:
                    edge.append(hit)
                elif not np.any(np.all(hit == edge, axis = 1)):
                    edge.append(hit)
                
        return np.array(edge)
        
    def get_clusters(self, conf):
        """ Label detached clusters and store size (not sorted yet) """
        self.visit = conf.copy()
        self.visit[conf == 0] = -1
        self.visit[conf == 1] = 0
        
        label = 0
        cluster_sizes = []
        while True:
            valid_starts = np.argwhere(self.visit == 0)
            # y, x = np.where(self.visit == 0)
            # valid_starts = np.array(list(zip(y, x)))
            if len(valid_starts) == 0: 
                break
            
            label += 1
            self.DFS(valid_starts[0], label)
        
            cluster_sizes.append(np.sum(self.visit == label))

        labels = [c for c in range(1, 1+len(cluster_sizes))]
        return labels, cluster_sizes
    
    def DFS(self, pos, label):
        """ Depth-first search (DFS) used for 
            detecting isolated clusters (walking on atoms not centers) """


        # Check if visited
        if self.visit[pos[0], pos[1]] != 0:
            return # Already visited
      
        # Mark as visited
        self.visit[pos[0], pos[1]] = label # Make dynamic labeling
            
        # Find potential neighbours
        neigh, _ = connected_neigh_atom(pos)
        on_sheet = np.all(np.logical_and(neigh < np.shape(self.visit), neigh >= (0,0)), axis = 1)
        neigh = neigh[on_sheet]
        
            
        # Start new search if neighbour atoms is present
        for pos in neigh:
            if self.visit[pos[0], pos[1]] == 0: # Atom is present
                self.DFS(pos, label)
                
      
      
    def get_top_string(self, topN, fmt = '0.4f'):
        s = '#--- Genetic algorithm --- #\n'
        s += f'Num. population = {self.A.shape[0]}\n'
        s += f'Generation = {self.gen}\n'
        s += f'Min score = {self.min_score:g}\n'
        s += f'Mean score = {self.mean_score:g}\n'
        s += f'Max score = {self.max_score:g}\n'
        s += f'porosity = {1-np.mean(self.A[0]):g}\n'
        s += f'Ellapsed time = {self.s_timing}\n'
        s += f'\n# Top {topN} scores \n'
        for top in range(np.min((topN, self.A.shape[0]))):
            s += f'top{top} | '
            s += f'{self.scores[top]:{fmt}}\n'
        
        return s          
    

    def print_top(self, topN, fmt = '0.4f'):
        print(self.get_top_string(topN, fmt))
      
      
        
    
    def save_top(self, save_path, topN = 5):
        filename = os.path.join(save_path, 'genetic_top.txt')

        try:
            outfile = open(filename, 'w')
        except FileNotFoundError:
            path = filename.split('/')
            os.makedirs(os.path.join(*path[:-1]))
            outfile = open(filename, 'w')
        
        # Write summary to file
        s = self.get_top_string(topN, fmt = '0.4f')
        outfile.write(s)
        outfile.close()
        
        # Save each top
        for top in range(np.min((topN, self.A.shape[0]))):
            name = f'top{top}'
            mat = self.A[top]
            np.save(os.path.join(save_path, name), mat)
            builder = config_builder(mat)
            builder.build()
            builder.save_view(save_path, 'sheet', name)
            
           
        
def porosity_target(conf):
    porosity = np.mean(conf)
    target_porosity = 0.2
    score = 1 - np.abs(porosity - target_porosity)
    return score

def ising_max(conf):
    score = 0
    Lx, Ly = conf.shape[0], conf.shape[1]
    for i in range(Lx):
        for j in range(Ly):
            set1 = [conf[i,j], conf[(i + 1 + Lx)%Lx,j]]
            set2 = [conf[i,j], conf[i, (j + 1 + Ly)%Ly]]
            score += np.sum(set1) + np.sum(set2) - 2*np.min(set1) - 2*np.min(set2)
    
    return score 





def run_pop_search(model_name, params, N = 50, num_generations = 50, topN = 5):
    """ Do genetic algorithm search based on a Tetrahedron pattern """
    GA = Genetic_algorithm(model_weights, model_info, N, image_shape = (62,106), repair = True)
    GA.stretch = np.linspace(0, 2, 100)
    GA.F_N = 5
    GA.set_fitness_func(GA.max_drop)
    
    
    name = f'pop_{params[0]}_{params[1]}_{params[2]}'
    size = (params[0], params[1])
    sp = params[2]
    population = []
    
    print(f"Creating population (N = {N}): {name}")
    for n in range(N):
        print(n)
        mat = pop_up(shape = (62, 106), size = size, sp = sp, ref = 'RAND')
        population.append(mat)
    GA.init_population(population)
    
    print(f'Running evolution for {num_generations} generations')
    GA.evolution(num_generations)
    
    print(f'Storing results (top {topN})')
    GA.print_top(topN)
    GA.save_top('./GA_{name}', topN)

def run_hon_search(model_name, params, N = 50, num_generations = 50, topN = 5):
    """ Do genetic algorithm search based on a Honeycomb pattern """
    GA = Genetic_algorithm(model_weights, model_info, N, image_shape = (62,106), repair = True)
    GA.stretch = np.linspace(0, 2, 100)
    GA.F_N = 5
    GA.set_fitness_func(GA.max_drop)
    
    
    xwidth = 2*params[0]- 1
    ywidth = params[1]
    bridge_thickness =  params[2]
    bridge_len = params[3]
    name = f'hon_{xwidth}_{ywidth}_{bridge_thickness}_{bridge_len}'
    population = []
    
    print(f"Creating population (N = {N}): {name}")
    for n in range(N):
        print(n)
        mat = honeycomb(shape = (62, 106), xwidth = xwidth, ywidth = ywidth, bridge_thickness = bridge_thickness, bridge_len = bridge_len, ref = 'RAND')
        population.append(mat)
    GA.init_population(population)
    print(f'Running evolution for {num_generations} generations')
    GA.evolution(num_generations)
    
    print(f'Storing results (top {topN})')
    GA.print_top(topN)
    GA.save_top(f'./GA_{name}', topN)
    
def run_RW_search(model_name, name, params, N = 50, num_generations = 50, topN = 5):
    """ Do genetic algorithm search based on a Random walk pattern """
    GA = Genetic_algorithm(model_weights, model_info, N, image_shape = (62,106), repair = True)
    GA.stretch = np.linspace(0, 2, 100)
    GA.F_N = 5
    GA.set_fitness_func(GA.max_drop)
    
    
    RW =  RW = RW_Generator(size = (62,106),
                            num_walks = params['num_walks'],
                            max_steps = params['max_steps'],
                            min_dis = params['min_dis'],
                            bias = params['bias'],
                            center_elem = params['center_elem'],
                            avoid_unvalid = params['avoid_unvalid'],
                            RN6 = params['RN6'],
                            grid_start = params['grid_start'],
                            centering = False,
                            stay_or_break = params['stay_or_break'],
                            avoid_clustering = 'repair', 
                            periodic = True
                    )
    
  
    population = []
    
    print(f"Creating population (N = {N}): {name}")
    for n in range(N):
        mat = RW.generate()
        population.append(mat)
    GA.init_population(population)
    
    print(f'Running evolution for {num_generations} generations')
    GA.evolution(num_generations)
    
    print(f'Storing results (top {topN})')
    GA.print_top(topN)
    GA.save_top(f'./GA_{name}', topN)
    
def run_porosity_search(model_name, name, porosity = [0.5], N = 50, num_generations = 50, topN = 5):
    """ Do genetic algorithm search from random noice of porosity """
    GA = Genetic_algorithm(model_weights, model_info, N, image_shape = (62,106), repair = True)
    GA.stretch = np.linspace(0, 2, 100)
    GA.F_N = 5
    GA.set_fitness_func(GA.max_drop)
    

    print(f"Creating population (N = {N}): {name}")
    GA.init_population(porosity)
    
    print(f'Running evolution for {num_generations} generations')
    GA.evolution(num_generations)
    
    print(f'Storing results (top {topN})')
    GA.print_top(topN)
    GA.save_top(f'./GA_{name}', topN)
    
    



if __name__ == '__main__': 
    model_name = 'mom_weight_search_cyclic/m0w0'
    model_weights = f'{model_name}/model_dict_state'
    model_info = f'{model_name}/best_scores.txt'
    
    
    
    # --- Genetic algorithm search: Max drop --- #
    N = 100
    num_generations = 50
    topN = 5
    
    
    # run_pop_search(model_name, (1,7,1), N, num_generations, topN)
    # run_hon_search(model_name, (3,3,5,3), N, num_generations, topN)
    
    # run_porosity_search(model_name, 'P05', [0.5], N, num_generations, topN)
    # run_porosity_search(model_name, 'P025', [0.25], N, num_generations, topN)
   
    # Top 5 RW for max drop     
    param1 = {'num_walks': 30, 'max_steps': 26, 'min_dis': 1, 'bias': [(0.03, -1.00), 9.23],  'center_elem': 'full', 'avoid_unvalid': False, 'RN6': False, 'grid_start': True, 'stay_or_break': 0.00} 
    param2 = {'num_walks': 18, 'max_steps': 20, 'min_dis': 2, 'bias': [(0.19, 0.98), 7.27],   'center_elem': 'full', 'avoid_unvalid': False, 'RN6': False, 'grid_start': True, 'stay_or_break': 0.00} 
    param3 = {'num_walks': 17, 'max_steps': 17, 'min_dis': 4, 'bias': [(-0.11, -0.99), 9.41], 'center_elem': 'full', 'avoid_unvalid': False, 'RN6': False, 'grid_start': True, 'stay_or_break': 0.00} 
    param4 = {'num_walks': 10, 'max_steps': 22, 'min_dis': 2, 'bias': [(0.23, 0.97), 7.38],   'center_elem': 'full', 'avoid_unvalid': True,  'RN6': False, 'grid_start': True, 'stay_or_break': 0.00} 
    param5 = {'num_walks': 17, 'max_steps': 5,  'min_dis': 4, 'bias': [(0.00, 1.00), 0.00],   'center_elem': False,  'avoid_unvalid': True,  'RN6': True,  'grid_start': True, 'stay_or_break': 0.64} 
   
    run_RW_search(model_name, 'RW1', param1, N, num_generations, topN)
    # run_RW_search(model_name, 'RW2', param2, N, num_generations, topN)
    # run_RW_search(model_name, 'RW3', param3, N, num_generations, topN)
    # run_RW_search(model_name, 'RW4', param4, N, num_generations, topN)
    # run_RW_search(model_name, 'RW5', param5, N, num_generations, topN)
    
    
    # ## TEST
    # GA = Genetic_algorithm(model_weights, model_info, N = 5, image_shape = (62,106), repair = False)
    # GA.stretch = np.linspace(0, 2, 100)
    # GA.F_N = 5
    # GA.set_fitness_func(GA.max_drop)
    # GA.init_population([0.01, 0.05, 0.1, 0.2, 0.3])
    # GA.evolution(num_generations = 1)
    
    # GA.print_top(topN)
    # GA.save_top('./GA_test', topN)
    
    # GA = Genetic_algorithm(model_weights, model_info, N = 50, image_shape = (10, 10), expand = (62,106), repair = True)
    # GA = Genetic_algorithm(model_weights, model_info, N = 10, image_shape = (10, 10), expand = None)
    
    # exit()
    # --- Define fitness --- #
    # GA.set_fitness_func(ising_max)
    # GA.init_population(['../config_builder/baseline/hon3215.npy', '../config_builder/baseline/pop1_7_5.npy', 0, 0.25, 0.5, 0.75, 1])
    # GA.init_population([0.25, 0.5, 0.75, 1])
    # GA.evolution(num_generations = 100)
    # GA.show_sheet(GA.A[0])
    # GA.show_sheet(GA.A_ex[0])
    # GA.show_status()
    # plt.show()
    
    
    
    
    # Initialize populartion
    # mat = np.zeros((5,10))
    # mat[0, 0] = 1
    # mat[0, 3] = 1
    # mat[0, 6] = 1
    # mat[0, 9] = 1
    # mat[1, 1] = 1
    # mat[1, 8] = 1
    # mat[2, 5] = 1
    # mat[2, 7] = 1
    # mat[2, 9] = 1
    # mat[3, 3] = 1
    # mat[4, 0:2] = 1
    # mat[4, 7] = 1
    # mat[4, 9] = 1
    # GA.init_population([mat])
  
  
    # np.random.seed(1235)
    # GA.init_population([0.35])
    # GA.show_sheet()
    # mat = GA.repair(GA.A[0])
    # GA.init_population([mat])
    # GA.show_sheet()
    # exit()
    # GA.get_clusters(mat)
    
    
    # GA.show_status()
    # plt.show()
    
    
    
    
    
    # plt.imshow(GA.A[0])
    # plt.show()
        


