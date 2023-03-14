from use_network import *

from config_builder.build_config import *



class Accelerated_search:
    def __init__(self, model_weights, model_info):
        
        # self.image_shape = (62, 106)
        self.image_shape = (10, 10)
        self.N = 100 # Population size
        self.A = np.zeros((self.N, *self.image_shape), dtype = int) # Population
        self.P = np.zeros((*self.image_shape, 2,2))
            
            
            # Transistion probabilities
        self.n = np.zeros((*self.image_shape, 2))
        self.n_target = np.zeros((*self.image_shape, 2))


        ### FILL RANDOMLY FOR NOW XXX
        # ones = np.random.rand(*np.shape(self.A)) < 0.5
        ones = np.random.rand(*np.shape(self.A)) < 0.5
        self.A[ones] = 1
        
        self.gen = 0
    
        self.scores = np.zeros(self.N)
    
    def fitness_func(self, conf):
        """ Tmp fitness function for testing """
        # Favorites certain porosity 
        
        score = 0
        Lx, Ly = self.image_shape[0], self.image_shape[1]
        for i in range(1, Lx):
            for j in range(1, Ly):
                x = (i + Lx)%Lx
                y = (i + Lx)%Lx
                set1 = [conf[i,j], conf[(i + 1 + Lx)%Lx,j]]
                set2 = [conf[i,j], conf[i,(j + 1 + Ly)%Ly]]
                score += np.sum(set1) + np.sum(set2) - 2*np.min(set1) - 2*np.min(set2)
        
        
        # porosity = np.mean(conf)
        # target_porosity = 0
        # score = 1 - np.abs(porosity - target_porosity)
        return score
    
    
    
    def evaluate_fitness(self):
        for i in range(self.N):
            self.scores[i] = self.fitness_func(self.A[i])
        self.rank = np.argsort(self.scores)[::-1] # In descending order
        
        self.scores = self.scores[self.rank]
        self.A = self.A[self.rank]
        # self.N_mark = self.N//10
        self.N_mark = self.N//10
        
        
        self.min_score  = self.scores[-1]
        self.mean_score = np.mean(self.scores)
        self.max_score  = self.scores[0]
        
      
        
    def update_state_distribution(self):
        C1 = np.mean(self.A, axis = 0)
        C0 = 1 - C1
        self.n[:, :, 0] = C0
        self.n[:, :, 1] = C1
        # TODO: Keep only C0 and n1 since these sum to one anyway 
        # and can be related as n1 = 1 - n0

    def update_state_distribution_target(self):
        C0_target = np.max(np.multiply(self.W[:, np.newaxis, np.newaxis], -self.A+1), axis = 0)
        C1_target = np.max(np.multiply(self.W[:, np.newaxis, np.newaxis], self.A), axis = 0)
        # TODO: What if the pixel is all zero or ones... XXX
        
        self.n_target[:, :, 0] = C0_target
        self.n_target[:, :, 1] = C1_target
     
        # Normalize
        self.n_target /= np.sum(self.n_target, axis = -1)[:, :, np.newaxis]
        
    def update_gene_transistion_probabilities(self):
        
        # --- Set P00 --- #
        if self.gen == 0:
            self.P[:, :, 0, 0] = 0.5
        else:
            self.P[:, :, 0, 0] = self.n[:, :, 0]
        
        
        # --- Calculate P10 --- #
        # Only apply formula to indexes for self.n[:, :, 1] != 0
        nonzero_n1 = self.n[:, :, 1] > 0
        self.P[nonzero_n1, 1, 0] = (self.n_target[nonzero_n1, 0] - self.P[nonzero_n1, 0, 0]*self.n[nonzero_n1, 0])/self.n[nonzero_n1, 1]
        self.P[~nonzero_n1, 1, 0] = 1
        
        
        
        
        prob_bound = 0
        # Clip the result at probability range [bound, 1-bound]
        underflow = self.P[:, :, 1, 0] < prob_bound
        overflow = self.P[:, :, 1, 0] > 1-prob_bound
        
        self.P[self.P[:, :, 1, 0] < prob_bound, 1, 0] = prob_bound
        self.P[self.P[:, :, 1, 0] > 1-prob_bound, 1, 0] = 1-prob_bound
        
        self.P[self.P[:, :, 0, 1] < prob_bound, 0, 1] = prob_bound
        self.P[self.P[:, :, 0, 1] > 1-prob_bound, 0, 1] = 1-prob_bound
        
    
        
        # # Watch a row
        # P10 = self.P[0, :, 1, 0]
        # P01 = self.P[0, :, 0, 1]
        # n0target = self.n_target[0, :, 0]
        # P00 = self.P[0, :, 0, 0]
        # n0 = self.n[0, :, 0]
        # n1 = self.n[0, :, 1]
        # pred_P10 = (n0target - P00*n0)/n1
        # print('P10     ', [f'{s:0.4f}' for s in P10])
        # print('P01     ', [f'{s:0.4f}' for s in P01])
        # print('n0target', [f'{s:0.4f}' for s in n0target])
        # print('P00     ', [f'{s:0.4f}' for s in P00])
        # print('n0      ', [f'{s:0.4f}' for s in n0])
        # print('n1      ', [f'{s:0.4f}' for s in n1])
        # print('pred_P10', [f'{s:0.4f}' for s in pred_P10])
        # print()
        
        # print(f'P10 = {np.mean(self.P[:, :, 1, 0]):g}, n0target = {np.mean(self.n_target[:, :, 0]):g}, P00 = {np.mean(self.P[:, :, 0, 0]):g}, n0 = {np.mean(self.n[:, :, 0]):g}, n1 = {np.mean(self.n[:, :, 1]):g}, avg P10 = {avg_P10}')
        
        # --- Calculate remaining P11 and P01 --- # XXX Never used explicitly 
        self.P[:, :, 1, 1] = 1 - self.P[:, :, 1, 0]
        self.P[:, :, 0, 1] = 1 - self.P[:, :, 0, 0]
        
        
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
        self.update_state_distribution()
        self.update_state_distribution_target()
        self.update_gene_transistion_probabilities()
        
        
        # XXX This might work, but then we need to introduce a similar
        # weighting in the target distirbution calculation XXX
        # i = np.arange(self.N).astype('float')
        # bias = np.where(i >= self.N_mark, len(i)-1-i, 1)
        # bias[self.N_mark:] /= bias[self.N_mark]
        for i in range(self.N):
           
            if mutate_individual[i]:
                RN = np.random.rand(*self.image_shape)
                zeros = self.A[i] < 0.5
                
                P01 = bias[i]*self.P[:, :, 0, 1] + (1-bias[i])*0.5
                P10 = bias[i]*self.P[:, :, 1, 0] + (1-bias[i])*0.5
                
                # flip0 = np.logical_and(RN < P01, zeros)
                # flip1 = np.logical_and(RN < P10, ~zeros)
                flip0 = np.logical_and(RN < self.P[:, :, 0, 1], zeros)
                flip1 = np.logical_and(RN < self.P[:, :, 1, 0], ~zeros)
                self.A[i][flip0] = 1
                self.A[i][flip1] = 0
        
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
                best_porosity = np.mean(self.A[0])
                
                print(f'Gen = {self.gen} | Min score = {self.min_score:g}, Mean score = {self.mean_score:g}, Max score = {self.max_score:g}, mean P01 = {np.mean(self.P[:, :, 0, 1]):g}, mean P10 = {np.mean(self.P[:, :, 1, 0]):g}')
                # print(f'Gen = {self.gen} | Max score = {self.max_score:g}, mean P01 = {np.mean(self.P[:, :, 0, 1]):g}, mean P10 = {np.mean(self.P[:, :, 1, 0]):g},  best porosity = {best_porosity:g}, avg td = {np.mean(self.n_target[:, :, 0]):g}, {np.mean(self.n_target[:, :, 1]):g}')
                
                # if self.gen % 100 == 0:
                #     plt.imshow(self.A[0])
                #     plt.show()
                # print(f'Gen = {self.gen} | Max score = {self.max_score:g}, mean P01 = {np.mean(self.P[:, :, 0, 1]):g}, mean P10 = {np.mean(self.P[:, :, 1, 0]):g},  best porosity = {best_porosity:g}, avg td = {np.mean(self.n_target[:, :, 0]):g}, {np.mean(self.n_target[:, :, 1]):g}')
                
                # print(f'Gen = {self.g
                # en} |, dist: best = {np.mean(self.n[:, :, 0]):g}, {np.mean(self.n[:, :, 1]):g}, target = {np.mean(self.n_target[:, :, 0]):g}, {np.mean(self.n_target[:, :, 1]):g}')
                self.evolve()
            except KeyboardInterrupt: 
                break
        
        
        
    # def evolution(self, num_generations = 20): 
    #     self.population_scores = self.get_scores(self.population)
    #     print(f'Gen = {self.gen} | N = {len(self.population)}, Min score = {np.min(self.population_scores):g}, Max score = {np.max(self.population_scores):g}')
    #     for generation in range(num_generations):
    #         try:
    #             self.evolve(num_mutations = 10, num_survivors = 2)
    #             print(f'Gen = {self.gen} | N = {len(self.population)}, Min score = {np.min(self.population_scores):g}, Max score = {np.max(self.population_scores):g}')
    #         except KeyboardInterrupt: 
    #             break
            
        
    def repair(self):
        # functionality to repair detached configurations
        pass
        
if __name__ == '__main__':
    name = 'graphene_h_BN/C16C32C64D64D32D16'
    model_weights = f'{name}_model_dict_state'
    model_info = f'{name}_best_scores.txt'
    
    # Input vals
    stretch = np.linspace(0, 2, 100)
    F_N = 5
    
    # Init
    AS = Accelerated_search(model_weights, model_info)
    AS.evolution()
    # AS.evaluate_fitness()
    # AS.mutate()