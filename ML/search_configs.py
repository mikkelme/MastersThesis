from accelerated_search import *



class Search:
    def __init__(self, model_path, topN, pattern):
        model_weights = os.path.join(model_path, 'model_dict_state')
        model_info = os.path.join(model_path, 'best_scores.txt')
        self.EV = Evaluater(model_weights, model_info)
        self.stretch = np.linspace(0, 2, 100)
        self.F_N = 5

        self.topN = topN

        self.shape = (62, 106)
        self.pattern = pattern
        
        self.extrema = {}
        self.extrema['Ff_min'] = np.array([['name', np.zeros(self.shape), 'stretch', 1e3] for n in range(topN)], dtype = 'object')
        self.extrema['Ff_max'] = np.array([['name', np.zeros(self.shape), 'stretch', 0] for n in range(topN)], dtype = 'object')
        self.extrema['Ff_max_diff'] = np.array([['name', np.zeros(self.shape), 'stretch_start', 'stretch_end', 0] for n in range(topN)], dtype = 'object')
        self.extrema['Ff_max_drop'] = np.array([['name', np.zeros(self.shape), 'stretch_start', 'stretch_end', 0] for n in range(topN)], dtype = 'object')
        self.patterns_evaluated = 0

        # print(self.extrema['Ff_max_drop'][0])
        # exit()
        
        # self.extrema = {'Ff_min': ['name', np.zeros(self.shape), 'stretch', 1e3],
        #                 'Ff_max': ['name', np.zeros(self.shape), 'stretch', 0],
        #                 'Ff_max_diff': ['name', np.zeros(self.shape), 'stretch_start', 'stretch_end', 0],
        #                 'Ff_max_drop': ['name', np.zeros(self.shape), 'stretch_start', 'stretch_end', 0]
        #                 }
       

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


    def insert(self, condition, name, mat, metrics, key):
        if np.any(condition): # Any hits
            insert_idx = np.argmax(condition)
            
            # Move over last values
            for i in reversed(range(insert_idx+1, self.topN)):
                self.extrema[key][i] = self.extrema[key][i-1]
                
            # Insert new values
            self.extrema[key][insert_idx][0] = name
            self.extrema[key][insert_idx][1] = mat
            self.extrema[key][insert_idx][2:] = metrics[key]
        

    def update_best(self, name, mat, metrics):
        # Minimum Ff
        condition = metrics['Ff_min'][-1] < self.extrema['Ff_min'][:, -1]
        self.insert(condition, name, mat, metrics, 'Ff_min')
        
        # Maximum Ff
        condition = metrics['Ff_max'][-1] > self.extrema['Ff_max'][:, -1]
        self.insert(condition, name, mat, metrics, 'Ff_max')
        
        # Maximum Ff diff
        condition = np.abs(metrics['Ff_max_diff'][-1]) > np.abs(self.extrema['Ff_max_diff'][:, -1])
        self.insert(condition, name, mat, metrics, 'Ff_max_diff')
        
        # Maximum Ff drop
        condition = metrics['Ff_max_drop'][-1] > self.extrema['Ff_max_drop'][:,-1]
        self.insert(condition, name, mat, metrics, 'Ff_max_drop')
        
   
        
        
        # if metrics['Ff_min'][-1] < self.extrema['Ff_min'][-1]:
        #     self.extrema['Ff_min'][0] = name 
        #     self.extrema['Ff_min'][1] = mat  
        #     self.extrema['Ff_min'][2] = metrics['Ff_min'][0] # stretch
        #     self.extrema['Ff_min'][3] = metrics['Ff_min'][1] # min Ff
            
        
        # # Maximum Ff
        # if metrics['Ff_max'][-1] > self.extrema['Ff_max'][-1]:
        #     self.extrema['Ff_max'][0] = name 
        #     self.extrema['Ff_max'][1] = mat  
        #     self.extrema['Ff_max'][2] = metrics['Ff_max'][0] # stretch
        #     self.extrema['Ff_max'][3] = metrics['Ff_max'][1] # max Ff
               
            
        # # Maximum Ff diff
        # if np.abs(metrics['Ff_max_diff'][-1]) > np.abs(self.extrema['Ff_max_diff'][-1]):
        #     self.extrema['Ff_max_diff'][0] = name 
        #     self.extrema['Ff_max_diff'][1] = mat  
        #     self.extrema['Ff_max_diff'][2] = metrics['Ff_max_diff'][0] # stretch start
        #     self.extrema['Ff_max_diff'][3] = metrics['Ff_max_diff'][1] # stretch end
        #     self.extrema['Ff_max_diff'][4] = metrics['Ff_max_diff'][2] # max diff (with sign)
            
        # # Maximum Ff drop
        # if metrics['Ff_max_drop'][-1] > self.extrema['Ff_max_drop'][-1]:
        #     self.extrema['Ff_max_drop'][0] = name 
        #     self.extrema['Ff_max_drop'][1] = mat  
        #     self.extrema['Ff_max_drop'][2] = metrics['Ff_max_drop'][0] # stretch start
        #     self.extrema['Ff_max_drop'][3] = metrics['Ff_max_drop'][1] # stretch end
        #     self.extrema['Ff_max_drop'][4] = metrics['Ff_max_drop'][2] # max drop
        
        
    def translate_input(self):
        pattern_name = self.pattern.__name__
        if pattern_name == 'honeycomb':
            name = f'{((1+self.current[0]//2), self.current[1], self.current[2], self.current[3])}' 
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
            
    
    
    def get_extrema_string(self, fmt = '0.2f'):
        s = f'Pattern = {self.pattern.__name__}\n'
        s += f'Max params = {self.max_params}\n'
        s += f'Top N = {self.topN}\n'
        s += f'Valid patterns = {self.patterns_evaluated}/{self.counter}\n'
        for key in self.extrema:
            s += f'\n# --- {key} --- #\n'
            for i in range(np.min((self.topN, self.patterns_evaluated))):
                s += f'{i} | name = {self.extrema[key][i, 0]} '
                for val in self.extrema[key][i, 2:]:
                    s += f'{val:{fmt}} '
                s += '\n'
                
        return s
       
    def print_extrema(self):
        print(self.get_extrema_string(fmt = '0.2f'))
      
        
    
    def save_extrema(self, save_path):
        filename = os.path.join(save_path, 'extrema.txt')

        try:
            outfile = open(filename, 'w')
        except FileNotFoundError:
            path = filename.split('/')
            os.makedirs(os.path.join(*path[:-1]))
            outfile = open(filename, 'w')
        
        
        s = self.get_extrema_string(fmt = '0.3f')
        outfile.write(s)
        
        for key in self.extrema:
            for top, mat in enumerate(self.extrema[key][:, 1]):
                np.save(os.path.join(save_path, f'{key}{top}_conf'), mat)
                
       
        # outfile.write(f'Pattern = {self.pattern.__name__}\n')
        # outfile.write(f'Max params = {self.max_params}\n')
        # for key in self.extrema:
        #     outfile.write(f'{key:11s} | ')
        #     for val in self.extrema[key]:
                
                
        #         if isinstance(val, (np.ndarray)):
        #             np.save(os.path.join(save_path, f'{key}_conf'), val)
        #         else:
        #             if isinstance(val, str):
        #                 outfile.write(f'{val} ')
        #             else:
        #                 outfile.write(f'{val:0.2f} ')
        #     outfile.write('\n')
                    
            
        
# TODO: Implement repeat functionality ?

if __name__ == '__main__':
    folder = 'training_2'
    model_name = f'{folder}/C16C32C64D64D32D16'
    
    
    S = Search(model_name, topN = 5, pattern = honeycomb)
    # S = Search(model_name, topN = 1, pattern = pop_up)
    # S = Search(model_name, topN = 1, pop_up)
    # S.search([9, 13, 4])
    
    # S.search([3, 5, 5, 5])
    S.search([1, 2, 2, 2])
    
    
    
    S.print_extrema()
    S.save_extrema('./extrema_folder')