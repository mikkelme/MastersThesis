import sys
sys.path.append('../') # parent folder: MastersThesis

from graphene_sheet.build_utils import *
import random

    

class RN_Generator:
    def __init__(self, size = (50, 70), num_walks = 10, max_steps = 30, max_dis = 10, bias = [(1, 0), 1], periodic = True, avoid_unvalid = False):

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
    
        # TODO: Implement center elem walks
        
        
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
            self.valid = self.add_dis_bound(del_map)
            
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
            
            p = self.get_p(direction)
            choice = np.random.choice(len(neigh), p = p)
            
            if available[choice] == False: # Hit unvalid site
                break
            
            pos = neigh[choice] 
            del_map.append(pos)
            self.valid[tuple(pos)] = 0.0
                    
        return np.array(del_map), self.valid


    def walk_dis(self, input, dis = 0, pre = []):
        """ Recursive function to walk to all sites
            within a distance of max_dis jumps """
            
        if self.max_dis == 0:
            return input
        
        for i, elem in enumerate(input):
            if isinstance(elem, (np.ndarray, np.generic)):
                input[i] = elem.tolist()


        neigh = []
        for pos in input:
            suggest = get_neighbour(pos)
            for s in suggest:
                if s not in pre and s not in neigh:
                    neigh.append(s)
            
        dis += 1
        if dis >= self.max_dis:
            return input + neigh
        else:
            pre = input
            return pre + self.walk_dis(neigh, dis, pre)
        

    def add_dis_bound(self, walk):
        m, n = np.shape(self.valid)
        for w in walk:
            new_del_map = np.array(self.walk_dis([w]))
            if self.periodic:
                 new_del_map = (new_del_map + (m,n))%(m,n)
            self.valid = delete_atoms(self.valid, new_del_map)
        return self.valid
    
    
    def get_p(self, input_dir):
        force_dir, strength = self.bias
    
        if strength == 0:
            return np.ones(len(input_dir))/len(input_dir)

        assert np.linalg.norm(force_dir) > 0, f"force direction {force_dir} has zero norm"

        # XXX: If strength == 1 ??
        norm = np.linalg.norm(input_dir, axis = 1)*np.linalg.norm(force_dir)
        dot = np.dot(input_dir, force_dir)/norm
        angle = np.where(np.abs(dot) >= 1, np.arccos(np.sign(dot)), np.arccos(dot))
        
        sigma = 10*np.exp(-4.6*strength)
        p = norm_dist(angle, sigma) / np.sum(norm_dist(angle, sigma))
        return p


    def grid_start(self):
        self.num_walks = 2
        L = int(np.ceil(np.sqrt(self.num_walks)))
        grid_idx = np.arange(L)
        # TODO make somehting like
        grid = [[0,0], [0,1], [1, 0], [1,1]] # Here for L = 2
        exit()
        
        x = []
        y = []
        for i in range(L):
            start = (i*self.size[0])//L
            stop = ((i+1)*self.size[0])//L 
            midpoint = start + (stop-start)//2
            x.append(midpoint)
       
            start = (i*self.size[1])//L
            stop = ((i+1)*self.size[1])//L 
            midpoint = start + (stop-start)//2
            y.append(midpoint)
       
        x, y = np,array(x), np.array(y)
        
        
        # choice = np.random.choice(grid, replace = False)
        
        
        
        # This is difficult and unessecary use of time...
        # num_walks = 1 => Put in center     
        # num_walks = 2 => put in each corner of a 2 x 2
        # num_walks = 3, 4 => fill up that 2 x 2
        # num_walks = 5 => make a 3 x 3 and fill cornes and the 5th in the middle (ideally)
        # print(self.size)
        
        
        

if __name__ == "__main__":
    
    RN = RN_Generator()
    RN.grid_start()
    # mat = RN.generate()
    
    
    