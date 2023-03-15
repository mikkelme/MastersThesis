from use_network import *

from config_builder.build_config import *

# matplotlib.interactive(True)

class Accelerated_search:
    # TODO: Simply stored matrixes by using n0 = 1 - n1 and similar for P matrx
    # TODO: Make initialization for kirigami dataset
    # TODO: Make functionality to use smaller populaiton and then translate it periodically to the whole sheet
    # TODO: Make repair function to ensure valid configurations once in a while or every generation perhaps. 
    def __init__(self, model_weights, model_info, N = 100, image_shape = (62, 106)):
        
        # self.image_shape = (62, 106)
        # self.image_shape = (10, 10)
        self.image_shape = image_shape
        self.N = N
        self.A = np.zeros((N, *image_shape), dtype = int) # Population
        
        self.EV = Evaluater(model_weights, model_info)
        
        # Transistion probabilities
        self.P = np.zeros((*self.image_shape, 2,2))
    
        # Distribution states
        self.n0 = np.zeros(self.image_shape)
        self.n0_target = np.zeros(self.image_shape)
        
        self.gen = 0
        
        self.scores = np.zeros(self.N)
    
        
    def init_population(self, configs):
        for i in range(self.N):
            conf = configs[i%len(configs)]
            if isinstance(conf, str): # Path to array
                self.A[i] = np.load(conf).astype(np.float32)
            elif isinstance(conf, np.ndarray): # Array
                self.A[i] = conf.copy()
            elif isinstance(conf, float): # Float defining site probability 
                ones = np.random.rand(*self.image_shape) < conf
                self.A[i][ones] = 1
        

    def set_fitness_func(self, func):
        self.fitness = func
        
        
    def max_drop(self, conf): 
        self.EV.set_config(conf)
        metrics = self.EV.evaluate_properties(self.stretch, self.F_N)
        score = metrics['max_drop'][-1] 
        return score
    

    # def fitness_func(self, conf):
    #     """ Tmp fitness function for testing """
    #     # Favorites certain porosity 
        
    #     # score = 0
    #     # Lx, Ly = self.image_shape[0], self.image_shape[1]
    #     # for i in range(Lx):
    #     #     for j in range(Ly):
    #     #         set1 = [conf[i,j], conf[(i + 1 + Lx)%Lx,j]]
    #     #         set2 = [conf[i,j], conf[i, (j + 1 + Ly)%Ly]]
    #     #         score += np.sum(set1) + np.sum(set2) - 2*np.min(set1) - 2*np.min(set2)
        
        
    #     porosity = np.mean(conf)
    #     target_porosity = 0.2
    #     score = 1 - np.abs(porosity - target_porosity)
    #     return score
        
    
    def evaluate_fitness(self):
        for i in range(self.N):
            self.scores[i] = self.fitness(self.A[i])
        self.rank = np.argsort(self.scores)[::-1] # In descending order
        
        self.scores = self.scores[self.rank]
        self.A = self.A[self.rank]
        # self.N_mark = self.N//2
        self.N_mark = self.N//10
        
        
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
            if mutate_individual[i]:
                RN = np.random.rand(*self.image_shape)
                zeros = self.A[i] < 0.5
                   
                flip0 = np.logical_and(RN < np.maximum(self.P[:, :, 0, 1], clip[i]), zeros)
                flip1 = np.logical_and(RN < np.maximum(self.P[:, :, 1, 0], clip[i]), ~zeros)
                # if i == self.N -1:
                #     print(f'flip0 = {np.sum(flip0)}, flip1 = {np.sum(flip1)}, clip = {clip[i]}')
                
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
                
                print(f'Gen = {self.gen} | Min score = {self.min_score:g}, Mean score = {self.mean_score:g}, Max score = {self.max_score:g}, mean P01 = {np.mean(self.P[:, :, 0, 1]):g}, mean P10 = {np.mean(self.P[:, :, 1, 0]):g}, porosity = {best_porosity:g}')
                
                # if self.gen % 100 == 1:
                #     self.show_status()
                #     plt.show()
                    
                    
                self.evolve()
            except KeyboardInterrupt: 
                break
        
    def show_status(self):
        fig, axes = plt.subplots(2, 2, num = unique_fignum(), figsize = (10, 5))
        axes[0,0].imshow(1 - self.n0, vmin = 0, vmax = 1)
        axes[0,0].set_title('1 -n0')
        axes[0,1].imshow(1- self.n0_target, vmin = 0, vmax = 1)
        axes[0,1].set_title('1 -n0 target')
        
        axes[1,0].imshow(self.P[:, :, 1, 0], vmin = 0, vmax = 1)
        axes[1,0].set_title('P10')
        
        axes[1,1].imshow(self.P[:, :, 0, 1], vmin = 0, vmax = 1)
        axes[1,1].set_title('P01')
        fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        
        
        fig_conf = plt.figure(num = unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
        plt.imshow(self.A[0], vmin = 0, vmax = 1)
        fig_conf.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
       
    def repair(self):
        # functionality to repair detached configurations
        pass
        
        
        
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
    AS = Accelerated_search(model_weights, model_info, N = 20, image_shape = (62,106))
    
    # Define fitness
    AS.stretch = np.linspace(0, 2, 100)
    AS.F_N = 5
    AS.set_fitness_func(AS.max_drop)
    
    # Initialize populartion
    AS.init_population([0.3])
    
    
    plt.imshow(AS.A[0])
    plt.show()
    # TODO: Try out repair function XXX
        

    
    # AS.evolution(num_generations = 10)
    # AS.show_status()
    # plt.show()