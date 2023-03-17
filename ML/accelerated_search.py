from use_network import *

from config_builder.build_config import *
from ase.visualize.plot import plot_atoms
from graphene_sheet.build_utils import *


# matplotlib.interactive(True)

class Accelerated_search:
    def __init__(self, model_weights, model_info, N = 100, image_shape = (62, 106), expand = None):

        # Settings        
        self.N = N
        self.image_shape = image_shape
        self.expand = expand
        
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
        score = metrics['max_drop'][-1] 
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
       
       
        mutate_individual = np.random.rand(self.N) < a
        
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

            if self.expand is not None:
                self.expand_population()    
                    
            # if mutate_individual[i]:
            #     zeros = self.A[i] < 0.5
                
            #     flip0 = np.logical_and(self.P[:, :, 0, 1] > np.quantile(self.P[:, :, 0, 1], 1 - a[i]), zeros)
            #     flip1 = np.logical_and(self.P[:, :, 1, 0] > np.quantile(self.P[:, :, 1, 0], 1 - a[i]), ~zeros)
               
            #     # RN = np.random.rand(*self.image_shape)
            #     # flip0 = np.logical_and(RN < np.maximum(self.P[:, :, 0, 1]*a[i], clip[i]), zeros)
            #     # flip1 = np.logical_and(RN < np.maximum(self.P[:, :, 1, 0]*a[i], clip[i]), ~zeros)
                
            #     # flip0 = np.logical_and(RN < np.maximum(self.P[:, :, 0, 1], clip[i]), zeros)
            #     # flip1 = np.logical_and(RN < np.maximum(self.P[:, :, 1, 0], clip[i]), ~zeros)
            #     # if i == self.N -1:
            #     #     print(f'flip0 = {np.sum(flip0)}, flip1 = {np.sum(flip1)}, clip = {clip[i]}')
                
            #     self.A[i][flip0] = 1
            #     self.A[i][flip1] = 0
        
        
    def evolve(self):
        """ Evolve by one generation """
        
        self.mutate()
        # self.repair()
        self.gen += 1
        self.evaluate_fitness()
        
        # No crossover for now
        # Mutate
        
    
    def evolution(self, num_generations = 1000):
        
        self.evaluate_fitness()
        for generation in range(num_generations):
            try:
                best_porosity = 1-np.mean(self.A[0])
                
                print(f'Gen = {self.gen} | Min score = {self.min_score:g}, Mean score = {self.mean_score:g}, Max score = {self.max_score:g}, mean P01 = {np.mean(self.P[:, :, 0, 1]):g}, mean P10 = {np.mean(self.P[:, :, 1, 0]):g}, porosity = {best_porosity:g}')
                
                # if self.gen % 10 == 0:
                #     fig = self.show_status()
                #     fig.savefig(f'AS/gen{self.gen}.pdf', bbox_inches='tight')
                #     # plt.show()
                #     plt.clf()
                    
                    # self.show_sheet()
                # if self.gen % 100 == 1:
                #     self.show_status()
                #     plt.show()
                    
                    
                self.evolve()
            except KeyboardInterrupt: 
                break
        
        
    def show_sheet(self):
        builder = config_builder(self.A[0])
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
            AS.EV.set_config(A)
            plot_atoms(builder.sheet, axes[0, 0], radii = 0.8, show_unit_cell = 0, scale = 1, offset = (0,0))
            AS.EV.stretch_profile(AS.stretch, AS.F_N, axes[0, 1])
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
    
       
    def repair(self, conf, max_walk_dis = None):
        # TODO: Add check for spanning cluster as well XXX
        # Repair by least changed atoms approah:
        # Try to add atom on the edge to connect until
        # the amount of added atoms surpasses the size of the cluster
        # then remoce the cluster instead.
        
        
        num_top_atoms = np.sum(conf[:, -1])
        num_bottom_atoms = np.sum(conf[:, 0])
        # If bottom or top row is already completely detached
        # then simply fill that row and let the walkers find
        # a connection (here unlimited walks will be provided)
        if num_top_atoms == 0:
            conf[:, -1] = 1
        if num_bottom_atoms == 0:
            conf[:, 0] = 1
            
        
        labels, cluster_sizes = self.get_clusters(conf)
        self.num_clusters = len(cluster_sizes)
        self.min_dis = 1 # For DFS
        # print(self.visit)
        # exit()
        
        # for label in reversed(labels):
        #     if label not in self.visit:
        #         continue
    
        while True:
            if len(labels) > 0:
                
                # Rearange labels to match cluster size
                # Biggest cluster first
                cluster_sizes = np.array(cluster_sizes)
                zero_map = np.argwhere(self.visit < 0)
                size_sort = np.argsort(cluster_sizes)[::-1]
                
                for i, from_label in enumerate(size_sort+1):
                    to_label = i+1
                    self.visit[self.visit == from_label] = -to_label
                self.visit = np.abs(self.visit)
                self.visit[zero_map[:, 0], zero_map[:, 1]] = -1
            
                labels = [c for c in range(1, 1+len(cluster_sizes))]
                cluster_sizes = list(cluster_sizes[size_sort])
                
                
                # resort labels by cluster size
                # sort = np.argsort(cluster_sizes)[::-1]
                # print(labels, cluster_sizes)
                # cluster_sizes = list(np.array(cluster_sizes)[sort])
                # labels = list(np.array(labels)[sort])
                # print(labels, cluster_sizes)
                label = labels[-1]
            else:
                break
            print('---')
            print(labels)
            print(self.visit)
            print(cluster_sizes)
            print('---')
    
            size = cluster_sizes[-1] # XXX
            if max_walk_dis is not None:
                size = max_walk_dis
            edge = self.get_edge(label) 
            
            self.visit_old = self.visit.copy()
            
            num_atoms_added = 0
            max_path_len = 0
            path = [[e] for e in edge]
            best_label = [-1 for e in edge]
            
            
            num_clusters = self.num_clusters
            while num_clusters > 1 and max_path_len <= size: 
                # for k in range(len(path)):
                #     print(path[k], best_label[k])
                # stop = input(f"while loop | label = {label}, size = {size}")
                for i in range(len(path)):
                    p = path[i]
                    
                    # Get current (end of path) position and label
                    current_pos = p[-1]
                    current_site_label = self.visit[current_pos[0], current_pos[1]]
                    
                    # Continue to next path if already on a
                    # non zero label not in its own cluster
                    if current_site_label > 0 and current_site_label != label:
                        best_label[i] = current_site_label
                        continue
                
                    
                    # Walk from end of path
                    walk = np.array(self.walk_dis([current_pos], label = label))
                    # Remove elements already in the preceding part of the path    
                    not_in_path = ~np.any(np.all(walk == np.array(p)[:, np.newaxis], axis = -1), axis = 0)
                    walk = walk[not_in_path]
                
                    # New walk destination site labels
                    site_labels = self.visit[walk[:, 0], walk[:, 1]]
                    
                    # Store all path combinations
                    if len(walk) > 0:
                        for w in range(1, len(walk)):
                            path.append(path[i] + [walk[w]])
                            best_label.append(site_labels[w])
                        path[i].append(walk[0])
                        best_label[i] = site_labels[0]
        
                    
                # Sort path by best label (1, 2, ...., max, -1)
                best_label_tmp = np.array(best_label)
                path_tmp = path.copy()
                sort_weight = np.where(best_label_tmp > 0, best_label_tmp, 1e3)
                label_sort = np.argsort(sort_weight)
                for idx_in, idx_out in enumerate(label_sort):
                    best_label[idx_in] = best_label_tmp[idx_out]
                    path[idx_in] = path_tmp[idx_out]
                
                
                # Check and remove duplicates on last site in path
                last_elements = [l[-1] for l in path]
                k = 0; k_end = len(path)
                while True:
                    match = np.all(path[k][-1] == last_elements, axis = 1)
                    match[k] = False
                    del_idx = np.argwhere(match).ravel()
                    for d in del_idx:
                        del path[d]
                        del last_elements[d]
                        del best_label[d]
                        k_end -= 1
                    k += 1
                    if k == k_end:
                        break
                
                # Update max path length 
                max_path_len = len(path[0]) + num_atoms_added
          
          
                # If any positive labels is present 
                if np.max(best_label) > 0:
                    # Go through paths and connect clusters with positive
                    # label hits until clusters is merged (if possible)  
                    sort_path = np.argsort(best_label)
                    already_merged = []
                    for s in sort_path:
                        p = np.array(path[s])
                        l = best_label[s]
                        if l < 0:
                            continue
                        
                        if l not in already_merged:
                            num_atoms_added += len(p) - 1 
                            max_path_len +=  len(p) - 1
                            
                            from_label = np.max((l, label))
                            to_label = np.min((l, label))
                            self.visit[p[:,0], p[:,1]] = to_label
                            self.visit[self.visit == from_label] = to_label
                            
                            # Merge clusters sizes
                            # Note: We only merge original sizes not the one from added atoms
                            from_idx = np.argmin(np.abs(from_label - labels))
                            to_idx = np.argmin(np.abs(to_label - labels))
                            cluster_sizes[to_idx] += cluster_sizes[from_idx] # XXX
                            cluster_sizes.pop(from_idx) 
                            # cluster_sizes[to_label-1] += cluster_sizes[from_label-1] # XXX
                            already_merged.append(to_label)
                            num_clusters -= 1
                            print(f'label = {label}, build: {p}, num_clusters: {self.num_clusters}->{num_clusters}')
                            
                            # Update labels
                            label = to_label # Upfate current label (Important for multiple walks from one label to multiple others)
                            labels.pop(from_idx) # Remove label from global list
                            
                            if not num_clusters > 1:
                                break
                            
                            if max_path_len > size+1:
                                break
                            
                    # Delete path and label list which is already merged
                    for idx in reversed(range(len(best_label))):
                        if best_label[idx] in already_merged:
                            best_label.pop(idx)
                            path.pop(idx)
                    
                    if len(path) == 0: # No more paths to build on
                        break
                            

                
                            
                if max_path_len > size:
                    # Break if progress is made
                    if num_clusters < self.num_clusters:
                        # print(f'num_clusters < self.num_clusters = {num_clusters < self.num_clusters}')
                        break
                    
                    # Check if the removing of the cluster would kill spanning possibilities
                    del_map = self.visit == label
                    
                    # Potential top and bottom atoms left when removing cluster
                    top_atoms_left = np.sum(self.visit[:, -1] > 0) - np.sum(del_map[:, -1])
                    bottom_atoms_left = np.sum(self.visit[:, 0] > 0) - np.sum(del_map[:, 0])
                    
                    # If this is killing spanning properties 
                    if top_atoms_left == 0 or bottom_atoms_left == 0: # Killing spanning properties
                        print(f"label = {label}, go nuts")
                        size += 1 # Unlimited walking allowed to find connection
            
            
                
            if not num_clusters > 1:
                # Repair completed
                break # break label loops
            
            if num_clusters == self.num_clusters: # Did not manage to reduce number of clusters
                # Remove cluster
                print(f'label {label}, removing cluster')
                self.visit = self.visit_old # Reset visit array
                self.visit[self.visit == label] = -1 # Delete label cluster
                num_clusters -= 1
            
                label_idx = np.argmin(np.abs(label - np.array(labels)))
                labels.pop(label_idx) # TODO: Thos can most likely be removed ...
                cluster_sizes.pop(label_idx) 
                

            self.num_clusters = num_clusters
            
          
        
        print(f'Repair completed') 

        conf[:] = 1
        conf[self.visit < 0] = 0
        return conf
    
    def get_edge(self, label):
        out = np.argwhere(self.visit == label)
        edge = []
        for pos in out:
            neigh, _ = connected_neigh_atom(pos)
            on_sheet = np.all(np.logical_and(neigh < np.shape(self.visit), neigh >= (0,0)), axis = 1)
            neigh = neigh[on_sheet]
            trial = self.visit[neigh[:, 0], neigh[:, 1]]
            out = np.argwhere(trial == -1).ravel()
            for hit in neigh[out]:
                if len(edge) == 0:
                    edge.append(hit)
                elif not np.any(np.all(hit == edge, axis = 1)):
                    edge.append(hit)
                
        return np.array(edge)
        
    
    def get_clusters(self, conf):
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
        # # Rearange labels to match cluster size
        # # Biggest cluster first
        # cluster_sizes = np.array(cluster_sizes)
        # zero_map = np.argwhere(self.visit < 0)
        # size_sort = np.argsort(cluster_sizes)[::-1]
        
        # for i, from_label in enumerate(size_sort+1):
        #     to_label = i+1
        #     self.visit[self.visit == from_label] = -to_label
        # self.visit = np.abs(self.visit)
        # self.visit[zero_map[:, 0], zero_map[:, 1]] = -1
        
        
        # labels = [c for c in range(1, 1+len(cluster_sizes))]
        # # np.arange(1, 1+len(cluster_sizes))
        # return labels, list(cluster_sizes[size_sort])
    
    
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



if __name__ == '__main__':
    
    # Initialize instance
    name = 'graphene_h_BN/C16C32C64D64D32D16'
    model_weights = f'{name}_model_dict_state'
    model_info = f'{name}_best_scores.txt'
    # AS = Accelerated_search(model_weights, model_info, N = 50, image_shape = (62,106))
    # AS = Accelerated_search(model_weights, model_info, N = 1, image_shape = (10, 10), expand = (62,106))
    AS = Accelerated_search(model_weights, model_info, N = 1, image_shape = (5, 10), expand = None)
    # AS = Accelerated_search(model_weights, model_info, N = 10, image_shape = (4, 4), expand = (100, 100))
    # AS = Accelerated_search(model_weights, model_info, N = 10, image_shape = (100, 100), expand =  None)
    
    # Define fitness
    AS.stretch = np.linspace(0, 2, 100)
    AS.F_N = 5
    # AS.set_fitness_func(AS.max_drop)
    # AS.set_fitness_func(AS.max_fric)
    AS.set_fitness_func(ising_max)
    
    # Initialize populartion
    # mat = np.zeros((5,10))
    # mat[:, :2] = 1
    # mat[0, 3] = 1
    # mat[1, 3] = 1
    # mat[0, 8] = 1
    # mat[1, 9] = 1
    # mat[2, 9] = 1
    # mat[4, 9] = 1
    # AS.init_population([mat])
  
    AS.init_population([0.7])
    AS.show_sheet()
    mat = AS.repair(AS.A[0])
    AS.init_population([mat])
    AS.show_sheet()
    exit()
    # AS.get_clusters(mat)
    
    exit()
    # TODO: Try out repair function on smaller image  XXX
    
    # AS.show_status()
    # plt.show()
    
    
    # AS.init_population(['../config_builder/baseline/hon3215.npy', '../config_builder/baseline/pop1_7_5.npy', 0, 0.25, 0.5, 0.74, 1])
    
    
    
    # plt.imshow(AS.A[0])
    # plt.show()
        



    # Get first stretch curve
    # AS.evolution(num_generations = 100)
    # AS.show_status()
    # plt.show()
    