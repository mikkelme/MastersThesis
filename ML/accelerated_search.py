from use_network import *

from config_builder.build_config import *
from ase.visualize.plot import plot_atoms
from graphene_sheet.build_utils import *


# matplotlib.interactive(True)

class Accelerated_search:
    # TODO: Make repair function to ensure valid configurations once in a while or every generation perhaps. 
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
            
                site_label = self.visit[s[0], s[1]]
                on_sheet =  np.all(np.logical_and(s < np.shape(self.visit), s >= (0,0)))
                if not s_in_pre and not s_in_neigh and site_label != label and on_sheet:
                        neigh.append(s)
            
        dis += 1
        if dis >= self.min_dis:
            return input + neigh
        else:
            pre = np.array(input)
            return  np.concatenate((pre, self.walk_dis(neigh, label, dis, pre)))
    
       
    def repair(self, conf):
        # Repair by least changed atoms approah:
        # Try to add atom on the edge to connect until
        # the amount of added atoms surpasses the size of the cluster
        # then remoce the cluster instead.
        labels, cluster_sizes = self.get_clusters(conf)
        for label in reversed(labels):
            if label not in self.visit:
                continue
            
    
            size = cluster_sizes[label-1]
            edge = self.get_edge(label) 
            
            # print(label, size)
            self.min_dis = 1
            path = [[e] for e in edge]
            best_label = [[-1] for e in edge]
            
            
            while True:
                for k in range(len(path)):
                    print(path[k], best_label[k])
                print()
                for i in range(len(path)):
                    p = path[i]
                    # Get current (end of path) position and label
                    current_pos = p[-1]
                    current_site_label = self.visit[current_pos[0], current_pos[1]]
                    
                    # Continue to next path if already on 
                    # non zero label not in its own cluster
                    if current_site_label > 0 and current_site_label != label:
                        best_label[i] = current_site_label
                        continue
                

                    # Walk from end of path
                    walk = np.array(self.walk_dis([current_pos], label = label))
                    
                    not_in_path = ~np.any(np.all(walk == np.array(p)[:, np.newaxis], axis = -1), axis = 0)
                    walk = walk[not_in_path]
                    
                    
                    # print("---")
                    # print(walk)
                    # print(path)
                    # print(walk[not_in_path])
                    # print("---")
                    
                    
                    # test = walk.copy()
                    # print('---')
                    # test_p = [np.array([1,2]), np.array((2,2))]
                    # # try:
                        
                    # #     test[2,1] = 1 
                    # #     test[2,0] = 2
                    # # except:
                    # #     pass
                    
                    
                    # print(test)
                    # print(test_p)
                    # # out = test == np.array(test_p)[:, np.newaxis]
                    # out = np.any(np.all(test == np.array(test_p)[:, np.newaxis], axis = -1), axis = 0)
                    # print(out)
                    # # print(p[0])
                    # # print(np.isin(test, [p[0]]))
                    # # print(~np.all(np.isin(test, p[0]), axis = 1))
                    # print('---')
                    # # walk = walk[~np.all(np.isin(walk, p), axis = 1)] # Remove sites already in current path
                    # # walk = walk[~np.all(np.isin(walk, np.array(path, dtype = object)), axis = 1)] # Remove sites already in path
                    
                    # # print('---')
                    # # # print(np.array(p))
                    # # # print('---')
                    # # print(np.array(path))
                    # # print('---')
                    # # print(walk)
                    # # print('---')
                    # # print(np.isin(walk, np.array(path, dtype = object)))
                    # # print('---')
                    
                    site_labels = self.visit[walk[:, 0], walk[:, 1]] # New site labels
                    
                    hits = site_labels > 0
                    if np.any(hits): # If any hits stop at the best (biggest cluster = lowest label)
                        exit("in here")
                        best = np.argmin(site_labels[hits])
                        test = site_labels[hits][best]
                        
                        path[i].append(walk[hits][best])
                        best_label[i] = site_labels[hits][best]
                        
                    else: # If all labels = -1 store all path combinations
                        # test = [l[-1] for l in path]
                        # for w in range(1, len(walk)):
                        #     # if not np.any(np.all(walk[w] == test, axis = 1)):
                        #     if w == 0:
                        #         path[i].append(walk[w])
                        #         best_label[i] = -1
                        #     else:
                        #         print('ap')
                        #         path.append(path[i] + [walk[w]])
                        #         best_label.append(-1)
                        
                        #     print(path)
                        #     print()
                        # exit()
                                
                        for w in range(1, len(walk)):
                            # print(path[i] + [walk[w]])
                            path.append(path[i] + [walk[w]])
                            best_label.append(-1)
                        path[i].append(walk[0])
                        best_label[i] = -1
                        
                print()
                # Check for duplicates on last site in path
                last_elements = [l[-1] for l in path]
                k = 0; k_end = len(path)
                while True:
                    print(path)
                    match = np.all(path[k][-1] == last_elements, axis = 1)
                    match[k] = False
                    del_idx = np.argwhere(match).ravel()
                    for d in del_idx:
                        print(d)
                        del path[d]
                        del last_elements[d]
                        k_end -= 1
                    
                    k += 1
                    if k == k_end:
                        break
                    
            
                        
                print()
                
                for k in range(len(path)):
                    print(path[k], best_label[k])
                exit()
                        
            # while True:
                
            #     hits = np.array(self.walk_dis(list(hits), label = label))
            #     print(self.visit[hits[:, 0], hits[:, 1]])
            #     print(hits)
            #     exit()
            
            # print(hits)
            # exit()
            # for e, pos in enumerate(edge):
            #     hits = np.array(self.walk_dis([pos], label = label))
            #     print(hits)
            #     exit()
            
            exit()
            
            
            best_pos = np.nan
            best_neigh = np.nan    # XXX
            lowest_label = 1e3 # XXX
            for e, pos in enumerate(edge):
                neigh, _ = connected_neigh_atom(pos)
                on_sheet = np.all(np.logical_and(neigh < np.shape(self.visit), neigh >= (0,0)), axis = 1)
                neigh = neigh[on_sheet]
                sites = self.visit[neigh[:,0], neigh[:,1]]
                best_label = np.min(sites, initial = 1e3, where = sites > 0) 
                if best_label < lowest_label:
                    best_pos = pos
                    best_neigh = neigh
                    lowest_label = int(best_label)
                
                if lowest_label == 1: # Just go for it TODO: 1 is not always the biggest cluster
                    break
            
            connecting_labels = self.visit[best_neigh[:,0], best_neigh[:,1]] # Watch out for -1 here
            merge = np.isin(self.visit, connecting_labels) # TODO: isin might not be safe as (x,y) = (y,x) = (x, x) = (y, y) generates a match... XXX
            self.visit[best_pos[0], best_pos[1]] = lowest_label
            self.visit[merge] = lowest_label
            print("flip", best_pos)

            if np.max(self.visit) < 1.5:
                break

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


        # Rearange labels to match cluster size
        # Biggest cluster first
        cluster_sizes = np.array(cluster_sizes)
        zero_map = np.argwhere(self.visit == -1)
        size_sort = np.argsort(cluster_sizes)[::-1]
        for i, to_label in enumerate(size_sort+1):
            from_label = i+1
            self.visit[self.visit == from_label] = -to_label
        self.visit = np.abs(self.visit)
        self.visit[zero_map[:, 0], zero_map[:, 1]] = -1
        
        labels = np.arange(1, 1+len(cluster_sizes))
        return labels, cluster_sizes[size_sort] 
    
    
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
    mat = np.ones((5,10))
    mat[0, 3] = 0
    mat[0, 4] = 0
    mat[1, 0] = 0
    mat[1, 2] = 0
    mat[1, 3] = 0
    mat[2, 0] = 0
    mat[2, 2] = 0
    
    # AS.init_population([mat])
    # AS.show_sheet()
    mat = AS.repair(mat)
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
    