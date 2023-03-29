from accelerated_search import *



class Search:
    def __init__(self, model_path):
        model_weights = os.path.join(model_path, 'model_dict_state')
        model_info = os.path.join(model_path, 'best_scores.txt')
        self.EV = Evaluater(model_weights, model_info)
        self.stretch = np.linspace(0, 2, 100)
        self.F_N = 5


        self.shape = (62, 106)
        self.pattern = None # Function for generating patterns
        self.extrema = {'Ff_min': ['name', np.zeros(self.shape), 'stretch', 1e3],
                        'Ff_max': ['name', np.zeros(self.shape), 'stretch', 0],
                        'Ff_max_diff': ['name', np.zeros(self.shape), 'stretch_start', 'stretch_end', 0],
                        'Ff_max_drop': ['name', np.zeros(self.shape), 'stretch_start', 'stretch_end', 0]
                        }
       
        self.patterns_evaluated = 0
        # self.best_configs = np.zeros((4, *self.shape))

    def get_next_combination(self):
        """ Get next combinaiton integer style """
        last = self.counter
        for j, p in enumerate(self.prod[1:]):
            self.current[j] = last // p
            last -= self.current[j] * p
        self.current[-1] = last%self.prod[-1]
        

    def evaluate(self, mat):
        self.EV.set_config(mat)
        metrics = self.EV.evaluate_properties(self.stretch, self.F_N)
        self.patterns_evaluated += 1
        return metrics

    def update_best(self, name, mat, metrics):
        # Minimum Ff
        if metrics['Ff_min'][-1] < self.extrema['Ff_min'][-1]:
            self.extrema['Ff_min'][0] = name 
            self.extrema['Ff_min'][1] = mat  
            self.extrema['Ff_min'][2] = metrics['Ff_min'][0] # stretch
            self.extrema['Ff_min'][3] = metrics['Ff_min'][1] # min Ff
            
        
        # Maximum Ff
        if metrics['Ff_max'][-1] > self.extrema['Ff_max'][-1]:
            self.extrema['Ff_max'][0] = name 
            self.extrema['Ff_max'][1] = mat  
            self.extrema['Ff_max'][2] = metrics['Ff_max'][0] # stretch
            self.extrema['Ff_max'][3] = metrics['Ff_max'][1] # max Ff
               
            
        # Maximum Ff diff
        if np.abs(metrics['Ff_max_diff'][-1]) > np.abs(self.extrema['Ff_max_diff'][-1]):
            self.extrema['Ff_max_diff'][0] = name 
            self.extrema['Ff_max_diff'][1] = mat  
            self.extrema['Ff_max_diff'][2] = metrics['Ff_max_diff'][0] # stretch start
            self.extrema['Ff_max_diff'][3] = metrics['Ff_max_diff'][1] # stretch end
            self.extrema['Ff_max_diff'][4] = metrics['Ff_max_diff'][2] # max diff (with sign)
            
        # Maximum Ff drop
        if metrics['Ff_max_drop'][-1] > self.extrema['Ff_max_drop'][-1]:
            self.extrema['Ff_max_drop'][0] = name 
            self.extrema['Ff_max_drop'][1] = mat  
            self.extrema['Ff_max_drop'][2] = metrics['Ff_max_drop'][0] # stretch start
            self.extrema['Ff_max_drop'][3] = metrics['Ff_max_drop'][1] # stretch end
            self.extrema['Ff_max_drop'][4] = metrics['Ff_max_drop'][2] # max drop
        
        
    def translate_input(self):
        pattern_name = self.pattern.__name__
        if pattern_name == 'honeycomb':
            name = str(self.current)
            return name, self.current
        if pattern_name == 'pop_up':
            size = (self.current[0], self.current[1])
            sp = self.current[2]
            name = str(self.current)
            return name, [size, sp]
        else:
            exit(f'\nPattern function {pattern_name} is not yet implemented.')

    def search(self, max_params = [1, 2, 2, 2]): # [3, 5, 5, 5]
        self.max_params = np.array(max_params) # xwidth, ywidth, bridge_thickness, bridge_len
        self.prod = [np.prod(self.max_params[p:]+1) for p in range(len(self.max_params))]
       
        # Go through all combinations [0, 0, ..., 0] --> max_params
        self.current = np.zeros(len(self.max_params), dtype = 'int')
        self.counter = 0
        for i in range(self.prod[0]):
            self.get_next_combination()
            print(f'\r{self.current} | ({self.patterns_evaluated}/{self.counter})         ', end = '')
            self.counter += 1
            
            try:
                # print(self.current)
                # exit()
                name, input = self.translate_input()
                mat = self.pattern(self.shape, *input, ref = None)
            except AssertionError: # Shape not allowed
                continue
            
            metrics = self.evaluate(mat)
            self.update_best(name, mat, metrics)
        print()
            
            
       
    def print_extrema(self):
        for key in self.extrema:
            vals = [self.extrema[key][i] for i in range(2, len(self.extrema[key]))]
            print(f'{key:11s} | name = {self.extrema[key][0]}, vals =  {vals}')
        
    
    def save_extrema(self, save_path):
        filename = os.path.join(save_path, 'extrema.txt')

        try:
            outfile = open(filename, 'w')
        except FileNotFoundError:
            path = filename.split('/')
            os.makedirs(os.path.join(*path[:-1]))
            outfile = open(filename, 'w')
        
        outfile.write(f'Pattern = {self.pattern.__name__}\n')
        outfile.write(f'Max params = {self.max_params}\n')
        for key in self.extrema:
            outfile.write(f'{key:11s} | ')
            for val in self.extrema[key]:
                
                
                if isinstance(val, (np.ndarray)):
                    np.save(os.path.join(save_path, f'{key}_conf'), val)
                else:
                    if isinstance(val, str):
                        outfile.write(f'{val} ')
                    else:
                        outfile.write(f'{val:0.2f} ')
            outfile.write('\n')
                    
            
        


if __name__ == '__main__':
    folder = 'training_2'
    model_name = f'{folder}/C16C32C64D64D32D16'
    
    
    S = Search(model_name)
    
    
    # S.pattern = honeycomb
    # S.search([3, 5, 5, 5])
    S.pattern = pop_up
    S.search([9, 13, 4])
    
    
    S.print_extrema()
    S.save_extrema('./extrema_folder')