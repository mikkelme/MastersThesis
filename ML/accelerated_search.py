from use_network import *

from config_builder.build_config import *



class Accelerated_search:
    def __init__(self, model_weights, model_info):
        
        # self.image_shape = (62, 106)
        self.image_shape = (60, 60)
        self.N = 10 # Population size
        self.N_mark = self.N//2
        self.A = np.zeros((10, *self.image_shape), dtype = int) # Population
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
        porosity = np.mean(conf)
        target_porosity = 0.2
        score = np.abs(porosity - target_porosity)
        return score
    
    
    
    def evaluate_fitness(self):
        for i in range(self.N):
            self.scores[i] = self.fitness_func(self.A[i])
            
        self.rank = np.argsort(self.scores)
        
        
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
        if self.gen == 0:
            self.P[:, :, 0, 0] = 0.5
        else:
            self.P[:, :, 0, 0] = self.n[:, :, 0]
        
        self.P[:, :, 1, 0] = 0
        map = self.n[:, :, 1] > 0
        self.P[map, 1, 0] = (self.n_target[map, 0] - self.P[map, 0, 0]*self.n[map, 0])/self.n[map, 1]
        
        
        # TODO: self.n[:, :, 1] = 0 gives trouble
        self.P[:, :, 1, 1] = 1 - self.P[:, :, 1, 0]
        self.P[:, :, 0, 1] = 1 - self.P[:, :, 0, 0]
        # TODO: Be carefull about orientation, in the math this is transposed...        

        test = [np.min(self.P), np.max(self.P)]
        print(test)
        # WORKING HERE
        # TODO: Can take values outside [0, 1] probability range....
        exit()
    def mutate(self):
        a = np.where(self.rank < self.N_mark, self.rank/self.N_mark, 1)
        self.W = 1 - a
        mutate_individual = np.random.rand(*np.shape(self.rank)) < a
        
        self.update_state_distribution()
        self.update_state_distribution_target()
        self.update_gene_transistion_probabilities()
        
        
        for i, rank in enumerate(self.rank):
            # print(i, mutate_row[i])
            if mutate_individual[i]:
                RN = np.random.rand(*self.image_shape)
                flip0 = RN < self.P[:, :, 0, 1]
                flip1 = RN < self.P[:, :, 1, 0]
                print(self.P[:, :, 1, 0])
                
                # print(RN)
                # print(self.A[i])
                exit()
        
    def evolve(self):
        """ Evolve by one generation """
        
        self.evaluate_fitness()
        # No crossover for now
        # Mutate
        
        
        self.gen += 1
        
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
    AS.evaluate_fitness()
    AS.mutate()