import sys
sys.path.append('../') # parent folder: MastersThesis

from graphene_sheet.RN_walks import *
from ML.accelerated_search import *

from scipy.stats import loguniform

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
        condition = metrics['Ff_min'][-1] <= self.extrema['Ff_min'][:, -1]
        self.insert(condition, name, mat, metrics, 'Ff_min')
        
        # Maximum Ff
        condition = metrics['Ff_max'][-1] >= self.extrema['Ff_max'][:, -1]
        self.insert(condition, name, mat, metrics, 'Ff_max')
        
        
        # print(name, np.argmax(self.extrema['Ff_max'][:, 0] == 'name'))
        
        
        
        # Maximum Ff diff
        condition = np.abs(metrics['Ff_max_diff'][-1]) >= np.abs(self.extrema['Ff_max_diff'][:, -1])
        self.insert(condition, name, mat, metrics, 'Ff_max_diff')
        
        # Maximum Ff drop
        condition = metrics['Ff_max_drop'][-1] >= self.extrema['Ff_max_drop'][:,-1]
        self.insert(condition, name, mat, metrics, 'Ff_max_drop')
        
        
    def translate_input(self):
        pattern_name = self.pattern.__name__
        if pattern_name == 'honeycomb':
            name = f'{((1+self.current[0]//2), self.current[1], self.current[2], self.current[3])}' 
            return name, self.current
        elif pattern_name == 'get_honeycomb_conf':
            name = self.pattern(None, *self.current, return_name = True)
            return name, self.current
        elif pattern_name == 'pop_up':
            size = (self.current[0], self.current[1])
            sp = self.current[2]
            name = str(self.current)
            return name, [size, sp]
        elif pattern_name == 'get_pop_up_conf':
            name = self.pattern(None, *self.current, return_name = True)
            return name, self.current
        elif pattern_name == 'RW_MC':
            name = 'RN'
            return name, self.current[:-1]
        elif pattern_name == 'get_RW_conf':
            name = self.pattern(None, *self.current, return_name = True)
            return name, self.current
        else:
            exit(f'\nPattern function {pattern_name} is not yet implemented.')

    def get_total_combinations(self, mp, sf):
        
        pattern_name = self.pattern.__name__
        if pattern_name == 'honeycomb':
            factors = [(mp[0]+1)//2 - (sf)//2, mp[1]+1-sf,(mp[2]+1)//2 - (sf)//2, (mp[3]+1)//2 - (sf)//2]
            return  np.prod(factors)
        elif pattern_name == 'pop_up':
            size_factor = 0
            for s0 in range(sf + 1 - sf%2, mp[0]+1, 2):
                for s1 in range(sf + 1 - sf%2, mp[1]+1, 2):
                    if (np.abs(s0 - s1) - 2)%4 == 0:
                        size_factor += 1
            return size_factor * mp[2]+1-sf
        else:
            return mp[-1]
        # elif pattern_name == 'RW_MC':
        # else:
        #     exit(f'\nPattern function {pattern_name} is not yet implemented.')

        

    def search(self, max_params = [1, 2, 2, 2], start_from = 1, repeat = 1): # [3, 5, 5, 5]
        self.max_params = np.array(max_params) - start_from
        self.current = np.zeros(len(self.max_params), dtype = 'int')
        total_comb = self.get_total_combinations(max_params, start_from)*repeat
        
        if self.pattern.__name__ == 'RW_MC':
            self.prod = [max_params[-1]]
            self.current[:-1] = self.max_params[:-1]
        else:
            self.prod = [np.prod(self.max_params[p:]+1) for p in range(len(self.max_params))]
        
    
    
        # Go through all combinations [0, 0, ..., 0] --> max_params
        self.counter = 0
        for i in range(self.prod[0]):
            try:
                self.get_next_combination()
                self.current += start_from
                self.counter += 1
                
                for r in range(repeat):
                    # print(f'\r{self.current} ({r+1}) | ({self.patterns_evaluated+1}/{total_comb})     ', end = '')
                    print(f'{self.current} ({r+1}) | ({self.patterns_evaluated+1}/{total_comb})')
                    try:
                        name, input = self.translate_input()
                        out = self.pattern(self.shape, *input, ref = 'RAND')
                        
                        if self.pattern.__name__ == 'RW_MC':
                            mat, RW = out
                            name = str(RW)[18:]
                        else:
                            mat = out
                        
                        assert mat is not None
                    except AssertionError: # Shape not allowed
                        continue
                    
                    
                    
                    metrics = self.evaluate(mat)
                
                    if metrics is not None:
                        self.update_best(name, mat, metrics)
                    # else:
                    #     print(name)
                    #     print('metrics is None')
                    #     exit()
                        
            
            except KeyboardInterrupt:
                break
        
        print()
                
    
    
    def get_extrema_string(self, fmt = '0.4f'):
        s = f'Pattern = {self.pattern.__name__}\n'
        s += f'Max params = {self.max_params}\n'
        s += f'Top N = {self.topN}\n'
        s += f'Valid patterns = {self.patterns_evaluated}/{self.counter}\n'
        for key in self.extrema:
            s += f'\n# --- {key} --- #\n'
            for i in range(np.min((self.topN, self.patterns_evaluated))):
                s += f'{i} | name = {self.extrema[key][i, 0]} '
                
                for val in self.extrema[key][i, 2:]:
                    # print(self.extrema[key][i, 0], self.extrema[key][i, 2:])
                    try:
                        s += f'{val:{fmt}} '
                    except ValueError:
                        print(f'error at key = {key}: {val}')
                s += '\n'
                
        return s
       
    def print_extrema(self):
        print(self.get_extrema_string(fmt = '0.4f'))
      
      
        
    
    def save_extrema(self, save_path):
        filename = os.path.join(save_path, 'extrema.txt')

        try:
            outfile = open(filename, 'w')
        except FileNotFoundError:
            path = filename.split('/')
            os.makedirs(os.path.join(*path[:-1]))
            outfile = open(filename, 'w')
        
        
        s = self.get_extrema_string(fmt = '0.4f')
        outfile.write(s)
        
        for key in self.extrema:
            for top, mat in enumerate(self.extrema[key][:, 1]):
                name = f'{key}{top}_conf'
                np.save(os.path.join(save_path, name), mat)
                builder = config_builder(mat)
                builder.build()
                builder.save_view(save_path, 'sheet', name)
                # dir, 'sheet', name, overwrite
                
      
            
        
# TODO: Implement repeat functionality ? XXX

def RW_MC(size, max_num_walks = 10, max_max_steps = 10, max_min_dis = 4, bias_max_temp = 10, ref = None):
    """ Random walk with monte carlo chosen parameters """
    
    # --- Get random sample --- #
    num_walks = random.randint(1, max_num_walks)
    max_steps = random.randint(1, max_max_steps)
    min_dis = random.randint(0, max_min_dis)
    
    RN_dir = np.random.uniform(0.0, 2*np.pi)
    RN_temp = random.choice([0, loguniform.rvs(0.1, bias_max_temp)])
    bias = [(np.cos(RN_dir), np.sin(RN_dir)), RN_temp]
    
    center_elem = random.choice(['full', False])
    avoid_unvalid = random.choice([True, False])
    RN6 = random.choice([True, False])
    grid_start = random.choice([True, False])
    # centering = np.random.choice([True, False]) #?? XXX
    stay_or_break = np.random.choice([np.random.uniform(0.0, 1.0), False])
    
    
    # --- Generate --- #
    RW = RW_Generator(size = size,
                        num_walks = num_walks,
                        max_steps = max_steps,
                        min_dis = min_dis,
                        bias = bias,
                        center_elem = center_elem,
                        avoid_unvalid = avoid_unvalid,
                        RN6 = RN6,
                        grid_start = grid_start,
                        centering = False,
                        stay_or_break = stay_or_break,
                        avoid_clustering = 'repair', 
                        periodic = True
                    )

    mat = RW.generate()
    return mat, RW



def get_pop_up_conf(shape, idx, return_name = False, ref = None):
    # shape, ref used as dummy variables to fit the format
    path = '../config_builder/popup' # len = 68
    filenames = get_files_in_folder(path, ext = '.npy')
    file = filenames[idx]
    n = file.strip('.npy').split('pop')[-1].replace('_','')
    name = f'({n[1]}, {n[2]}, {n[0]})'
    if return_name:
        return name
    else:
        mat = np.load(file)
        return mat
    
def get_honeycomb_conf(shape, idx, return_name = False, ref = None):
    # shape, ref used as dummy variables to fit the format
    path = '../config_builder/honeycomb' # len = 45
    filenames = get_files_in_folder(path, ext = '.npy')
    file = filenames[idx]
    n = file.strip('.npy').split('hon')[-1]
    name = f'{((1+int(n[0])//2), int(n[1]), int(n[2]), int(n[3]))}'     
    if return_name:
        return name
    else:
        mat = np.load(file)
        return mat
    
def get_RW_conf(shape, idx, return_name = False, ref = None):
    # shape, ref used as dummy variables to fit the format
    path = '../config_builder/RW' # len = 100
    filenames = get_files_in_folder(path, ext = '.npy')
    file = filenames[idx]
    name = file.strip('.npy').split('/')[-1]
    if return_name:
        return name
    else:
        mat = np.load(file)
        return mat


if __name__ == '__main__':
    model_name = 'mom_weight_search_cyclic/m0w0'
    topN = 50
    
    
    # --- Test against data --- #
    # Pop up
    # S = Search(model_name, topN = 50, pattern = get_pop_up_conf)
    # S.search([68-1], start_from = 0) 
    # S.print_extrema()
    
    # Honeycomb
    # S = Search(model_name, topN = 45, pattern = get_honeycomb_conf)
    # S.search([45-1], start_from = 0) 
    # S.print_extrema()
  
    # RW
    # S = Search(model_name, topN = 100, pattern = get_RW_conf)
    # S.search([100-1], start_from = 0) 
    # S.print_extrema()
    # S.extrema['Ff_max']
    
    
    # --- Extended search --- #
    # Pop up
    S = Search(model_name, topN, pattern = pop_up)
    S.search([60, 60, 30], start_from = 1, repeat = 10) # XXX
    S.print_extrema()
    S.save_extrema('./pop_search')
    
    # Honeycomb
    # S = Search(model_name, topN, pattern = honeycomb)
    # S.search([30, 30, 30, 60], start_from = 1, repeat = 10) # XXX
    # S.print_extrema()
    # S.save_extrema('./hon_search')
    
    # Random walk
    # S = Search(model_name, topN, pattern = RW_MC)
    # S.search([30, 30, 4, 10, int(1e5)], start_from = 0) # XXX
    # S.print_extrema()
    # S.save_extrema('./RW_search')
    
        