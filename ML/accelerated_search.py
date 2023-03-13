from use_network import *

from config_builder.build_config import *



class Accelerated_search:
    def __init__(self, model_weights, model_info):
        
        # self.image_shape = (62, 106)
        self.image_shape = (5, 7)
        self.N = 10 # Population size
        self.N_mark = self.N//2
        self.A = np.zeros((10, *self.image_shape), dtype = int) # Population
        self.n = np.zeros((*np.shape(self.A), 2))


        ### FILL RANDOMLY FOR NOW XXX
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
        
        

    def mutate(self):
        # print(self.scores)
        # print(self.scores[self.rank])
        # a = np.where(self.rank < self.N_mark, self.rank/self.N_mark, self.rank/self.N_mark)
         
        a = np.where(self.rank < self.N_mark, self.rank/self.N_mark, 1)
        s = 1 - a
        w = s
        ### TODO: Working here 
        
        
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