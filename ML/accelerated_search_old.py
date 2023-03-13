from use_network import *

from config_builder.build_config import *



class Genetic_algorithm:
    def __init__(self, model_weights, model_info):
        self.model_weights = model_weights
        self.model_info = model_info
        self.EV = Evaluater(self.model_weights, self.model_info)
        self.gen = 0
        self.set_mutator()
    def set_population(self, population):
        """ Set population from config path or numpy matrix input """
        self.population = []
        for pop in population:
            if isinstance(pop, str):
                self.population.append(self.load_config(pop)) 
            elif isinstance(pop, np.ndarray):
                self.population.append(pop)
            else:
                exit(f'population input of type {type(pop)} is not accepted')
                
                
    def set_input(self, stretch, F_N):
        self.stretch = stretch 
        self.F_N = F_N
        
    def load_config(self, config_path):
        """ Load configuration as numpy matrix """
        return np.load(config_path).astype(np.float32)
        

    def set_mutator(self): # TODO: Make generic
        self.mutate = self.uniform_flip
        # self.mutate = self.put_square

    def put_square(self, mat):
        pass
        # draw random coordinate and put square
        # Put all to 0 or all to 1 by some chance
        # Let size of square vary

    def uniform_flip(self, mat):
        # TODO: Organize framework for different mutator option
        # XXX One version: Flip by change p
        p = 0.005
        flip = np.random.rand(*np.shape(mat)) < p
        new_mat = mat.copy()
        new_mat[flip] += -2*new_mat[flip]+1
        return new_mat
        
    def generate_mutations(self, num_mutations):
        self.mutations = []
        # self.mutation_scores = []
        for gene in self.population:
            for m in range(num_mutations):
                mat = self.mutate(gene)
                self.mutations.append(mat)
    
    
    def get_scores(self, population):
        scores = np.zeros(len(population))
        for i, mat in enumerate(population):
            self.EV.set_config(mat)
            metrics = self.EV.evaluate_properties(self.stretch, self.F_N)
            # scores[i] = metrics['max_drop'][-1] # TODO: Make generic (this is hardcoded to forward drop)
            scores[i] = metrics['Ff_max'][-1] # XXX
        return scores
    
    def pick_top_genes(self, population, num_survivors, mode = None):
        # TODO: Make flexible for different metrics, low/high friction and 
        # max drop. Maybe make mapping from value to a score such that we
        # always maximize or minimize the score.
        
        scores = self.get_scores(population)
        argsort = np.argsort(scores)
        top_idx = argsort[-num_survivors:]
        
        # Pick the highest score as default so far XXX
        new_population = [population[i] for i in top_idx]
        return new_population, scores[top_idx]
        
    def evolve(self, num_mutations, num_survivors):
        # TODO: Maybe define num_mutations in a absolut manner instead of mutations per population ...? XXX
        self.generate_mutations(num_mutations)
        self.population += self.mutations # XXX This might not be the traditional way
        self.population, self.population_scores = self.pick_top_genes(self.population, num_survivors)
        self.gen += 1
            
    
    def evolution(self, num_generations = 20): 
        self.population_scores = self.get_scores(self.population)
        print(f'Gen = {self.gen} | N = {len(self.population)}, Min score = {np.min(self.population_scores):g}, Max score = {np.max(self.population_scores):g}')
        for generation in range(num_generations):
            try:
                self.evolve(num_mutations = 10, num_survivors = 2)
                print(f'Gen = {self.gen} | N = {len(self.population)}, Min score = {np.min(self.population_scores):g}, Max score = {np.max(self.population_scores):g}')
            except KeyboardInterrupt: 
                break
            
    
        # top_population, top_population_scores = self.pick_top_genes(self.population, num_survivors = 1)
        # for mat in top_population:
        #     self.show_config(mat)
            # print(scores)
        # test = self.mutations[0]
        # self.show_config()
        
        
    
    def show_config(self, mat):
        builder = config_builder(mat)
        builder.view()

        
        # metrics = EV.evaluate_properties(show = False)
        # print(metrics)
        
        
        
if __name__ == '__main__':
    name = 'graphene_h_BN/C16C32C64D64D32D16'
    model_weights = f'{name}_model_dict_state'
    model_info = f'{name}_best_scores.txt'
    
    # Input vals
    stretch = np.linspace(0, 2, 100)
    F_N = 5
    
    # Init
    GA = Genetic_algorithm(model_weights, model_info)
    GA.set_input(stretch, F_N)
    
    mat = np.ones((62, 106))
    # mat[28:32, 20: 80] = 0
    # mat[10:40, 50: 60] = 0
    
    GA.set_population(['../config_builder/baseline/hon3215.npy'])
    GA.set_population([mat])
    
    # Get first stretch curve
    GA.EV.set_config(GA.population[0])
    GA.EV.stretch_profile(GA.stretch, GA.F_N)
    
    # Evolve
    GA.evolution(num_generations = 200)
    top_population, top_population_scores = GA.pick_top_genes(GA.population, num_survivors = 1)
    top_mat = top_population[0]
    GA.show_config(top_mat)
    
    # Get final best stretch curve
    GA.EV.set_config(top_mat)
    GA.EV.stretch_profile(GA.stretch, GA.F_N)
    
    # show
    plt.show()
    
    
    # AS.load_config(config_path = '../config_builder/baseline/hon3215.npy')
    
    # AS.genetic_algorithm()
    
    # AS.show_config(AS.mat)
    # mat = AS.mutator(AS.mat)
    # AS.show_config(mat)
    