import sys
sys.path.append('../') # parent folder: MastersThesis

from graphene_sheet.build_utils import *
import random

    

class RN_Generator:
    def __init__(self, size = (50, 70), num_walks = 100, max_steps = 30, max_dis = 20, bias = [(2, 1), 1], periodic = False, avoid_unvalid = False):

        size = (4,10)
        ##############################
        
        self.size = np.array(size)
        self.num_walks = num_walks
        self.max_steps = max_steps
        self.max_dis = max_dis
        self.bias = bias
        self.periodic = periodic
        self.avoid_unvalid = avoid_unvalid
        
        self.mat = np.ones(size)    # lattice matrix
        self.valid = np.ones(size)  # valid positions
    
    
        if self.periodic:
            assert np.all(self.size%2 == 0), f"The size of the sheet {self.size} must have even side lengths to enable periodic boundaries."
    
        # TODO: Consider using the proper random generator
        #       suggested by numpy.
    
    def generate(self):        
        for w in range(self.num_walks):
            idx = np.argwhere(self.valid == 1)
            if len(idx) == 0:
                break
            
            start = random.choice(idx)
            del_map, self.valid = self.walk(start)
            self.mat = delete_atoms(self.mat, del_map)
            self.valid = add_dis_bound(del_map, self.valid, self.max_dis)
            
        return self.mat
        


    def walk(self, start):
        self.valid[tuple(start)] = 0
        
        pos = start
        del_map = []
        for i in range(self.max_steps):
            neigh, direction = connected_neigh(pos)
            m, n = np.shape(self.mat)   
            
            if self.periodic: 
                neigh = (neigh + (m,n))%(m,n)
            
            on_sheet = np.all(np.logical_and(neigh < (m,n), neigh >= (0,0)), axis = 1)
            idx = np.argwhere(on_sheet)[:,0]
            available = on_sheet
            available[idx] = self.valid[neigh[on_sheet][:,0], neigh[on_sheet][:,1]] == 1
        
            if self.avoid_unvalid:
                neigh = neigh[available]
                direction = direction[available]
                available = available[available]
                if len(neigh) == 0: # No where to go
                    break
            
            p = get_p(direction, np.array(self.bias[0]), self.bias[1])
            choice = np.random.choice(len(neigh), p = p)
            
            
            if available[choice] == False: # Hit unvalid site
                break
            
    
            pos = neigh[choice] 
            del_map.append(pos)
            self.valid[tuple(pos)] = 0.0
                    
        return np.array(del_map), self.valid








if __name__ == "__main__":
    
    RN = RN_Generator()
    mat = RN.generate()
    
    
    