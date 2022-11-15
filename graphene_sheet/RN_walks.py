import sys
sys.path.append('../') # parent folder: MastersThesis

from graphene_sheet.build_utils import *
import random

    

class RN_Generator:
    def __init__(self, size = (50, 70), num_walks = 1, max_steps = 0, max_dis = 0, bias = [(1, 1), 0], periodic = False, avoid_unvalid = False, grid_start = False, center_elem = True):

        size = (4,10)
        ##############################

        shape_error = f"SHAPE ERROR: Got size {size}, y-axis must be multiple of 2 and both nonzero positive integers."
        assert size[0]%1 == 0 and size[1]%1 == 0 and size[1]%2 == 0, shape_error
        assert size[0] > 0 and size[1] > 0, shape_error
        
        self.size = np.array(size)
        self.num_walks = num_walks
        self.max_steps = max_steps
        self.max_dis = max_dis
        self.bias = bias
        self.periodic = periodic
        self.avoid_unvalid = avoid_unvalid
        self.grid_start = grid_start
        self.center_elem = True
        
        self.mat = np.ones(size)    # lattice matrix
        self.valid = np.ones(size)  # valid positions
    
    
        if self.center_elem:
            center_size = (int(size[0] + 1), int(size[1]//2)) # TODO: Double check 
            print(center_size)
            exit()

    
        if self.periodic:
            assert np.all(self.size%2 == 0), f"The size of the sheet {self.size} must have even side lengths to enable periodic boundaries."
    
        # TODO: Implement center elem walks
        # TODO: Consider using the proper random generator
        #       suggested by numpy.
        
    
    def generate(self):        
        for w in range(self.num_walks):
            if self.grid_start:
                idx = self.get_grid()
            else:
                idx = np.argwhere(self.valid == 1)
            if len(idx) == 0:
                break
            
            start = random.choice(idx)
            del_map, self.valid = self.walk(start)
            
            self.mat = delete_atoms(self.mat, center_elem_trans_to_atoms(del_map, full = True)) # TODO Full = True/False
            
            
            self.mat = delete_atoms(self.mat, del_map)
            self.valid = self.add_dis_bound(del_map)
            
        return self.mat
        

    def walk(self, start):
        self.valid[tuple(start)] = 0
        del_map = [start]
        
        
        pos = start
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
            else:
                map = []
                for i in range(len(neigh)):
                    map.append(~np.any(np.all(neigh[i] == del_map, axis = 1)))
                neigh = neigh[map]
                direction = direction[map]
                available = available[map]
                
            
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


    def get_grid(self):
        L = int(np.ceil(np.sqrt(self.num_walks)))
     
        grid = []
        for i in range(L):
            xstart = (i*self.size[0])//L
            xstop = ((i+1)*self.size[0])//L 
            xpoint = xstart + (xstop-xstart)//2
            for j in range(L):
                ystart = (j*self.size[1])//L
                ystop = ((j+1)*self.size[1])//L 
                ypoint = ystart + (ystop-ystart)//2
                grid.append([xpoint, ypoint])
                
        grid = np.array(grid)
        idx = grid[self.valid[grid[:, 0], grid[:,1]] == 1]
        return idx
        
        

if __name__ == "__main__":
    
    RN = RN_Generator()
    # mat = RN.generate()
    
    
    