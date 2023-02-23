import sys
sys.path.append('../') # parent folder: MastersThesis
sys.setrecursionlimit(10000)

from graphene_sheet.build_utils import *
import random


class RW_Generator:
    def __init__(self,  size = (62, 106),
                        num_walks = 9,
                        max_steps = 6,
                        min_dis = 2,
                        bias = [(1, 1), 1],
                        RN6 = True,
                        periodic = True,
                        avoid_unvalid = False,
                        grid_start = True,
                        center_elem = True,
                        avoid_clustering = 10,
                        center = False):


        shape_error = f"SHAPE ERROR: Got size {size}, y-axis must be multiple of 2 and both nonzero positive integers."
        assert size[0]%1 == 0 and size[1]%1 == 0 and size[1]%2 == 0, shape_error
        assert size[0] > 0 and size[1] > 0, shape_error
        
        self.size = np.array(size)
        self.num_walks = num_walks
        self.max_steps = max_steps
        self.min_dis = min_dis
        self.bias = bias
        self.RN6 = RN6
        self.periodic = periodic
        self.avoid_unvalid = avoid_unvalid
        self.grid_start = grid_start
        self.center_elem = center_elem
        self.avoid_clustering = avoid_clustering
        self.center = center # Move CM as close to starting point as possible
        self.initialize()
        
    def initialize(self):
        if self.center_elem:
            center_size = np.array((int(self.size[0] + 1), int(self.size[1]//2))) # TODO: Double check 
            self.mat = np.ones(center_size, dtype = int)    # lattice matrix
            self.valid = np.ones(center_size, dtype = int)  # valid positions
            self.connected_neigh = connected_neigh_center_elem
            
        else:
            self.mat = np.ones(self.size, dtype = int)    # lattice matrix
            self.valid = np.ones(self.size, dtype = int)  # valid positions
            self.connected_neigh = connected_neigh_atom
    
    

    
        if self.periodic:
            assert np.all(self.size%2 == 0), f"The size of the sheet {self.size} must have even side lengths to enable periodic boundaries."
    
        # TODO: Work on single walk copied to multiple locations
        
        # TODO: Distributions on RN walk length?
    
        # TODO: Consider using the proper random generator
        #       suggested by numpy.
        
    

    
    def generate(self):        
        for w in range(self.num_walks):
            if self.grid_start:
                idx = self.get_grid()
                if len(idx) == 0: break
                start = idx[0]
            else:
                idx = np.argwhere(self.valid == 1)
                if len(idx) == 0: break
                start = random.choice(idx)
                
                
            del_map, self.valid = self.walk(start)
            if self.center:
                self.center_walk(del_map)
                
            self.mat = delete_atoms(self.mat, del_map)
            self.valid = self.add_dis_bound(del_map) 
        
        if self.center_elem: # transform from center elements to atoms
            del_map = np.column_stack((np.where(self.mat == 0)))
            del_map = center_elem_trans_to_atoms(del_map, full = True) # TODO: Toggle full on/off and handle problem with periodic boundary conditions
            if self.periodic and len(del_map) > 0:
                m, n = self.size
                del_map = (del_map + (m,n))%(m,n)
            self.mat = np.ones(self.size)
            self.mat = delete_atoms(self.mat, del_map)
            
            # # XXX For testing XXX
            # del_map = np.column_stack((np.where(self.valid == 0)))
            # del_map = center_elem_trans_to_atoms(del_map, full = True)
            # if self.periodic:
            #     m, n = self.size
            #     del_map = (del_map + (m,n))%(m,n)
            # self.valid = np.ones(self.size)
            # self.valid = delete_atoms(self.valid, del_map)
        
        # Avoid isolated clusters
        if self.avoid_clustering is not None:
            # Create binary matrix for visited sites (1 = visited)
            self.visit = np.zeros(self.size, dtype = int)
            
            # Walk configuration
            self.DFS((0,0))
            
            # Check if all sites are visited
            detect = np.sum(self.mat - self.visit) > 0.5
            if detect: # Isolated cluster detected
                self.avoid_clustering -= 1
                print(f'Isolated cluster detected | {self.avoid_clustering} attempts left')
                if self.avoid_clustering > 0:
                    self.initialize()
                    self.generate()
                else:
                    print('Removing isolated clusters')
                    self.mat = self.visit.copy()
            
        # Returns multiple times, but all with the correct matrix though...
        # Not sure if this is a problem XXX
        return self.mat
            
   

    def DFS(self, pos):
        """ Depth-first search (DFS) used for 
            detecting isolated clusters """

        # Check is visited
        if self.visit[pos[0], pos[1]] == 1:
            return # Already visited
      
        # Mark as visited
        self.visit[pos[0], pos[1]] = 1
            
        # Find potential neighbours (with PB)
        neigh, _ = self.connected_neigh(pos)
        neigh = (neigh + self.size)%self.size # PB
        
        # Start new search if neighbour atoms is present
        for pos in neigh:
            if self.mat[pos[0], pos[1]] == 1: # Atom is present
                self.DFS(pos)
                

    def center_walk(self, del_map):
        
        start = del_map[0] # Starting point
        CM = np.round(np.mean(del_map, axis = 0)) # Approximate CM
        
        # Move towards CM and check if valid
        exit()
    

    def walk(self, start):
        self.valid[tuple(start)] = 0
        del_map = [start]
        
        if self.RN6:
            six_directions = [(0,1), (1,1), (1,-1), (0,-1), (-1,-1), (-1,1)]
            force_dir = six_directions[np.random.choice(len(six_directions), 1)[0]]
            self.bias[0] = force_dir
      
        
        pos = start    
        for i in range(self.max_steps):
            neigh, direction = self.connected_neigh(pos)
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
                
            if len(neigh) == 0:
                break
            
            p = self.get_p(direction)
            choice = np.random.choice(len(neigh), p = p)
            
            if available[choice] == False: # Hit unvalid site
                break
            
            pos = neigh[choice] 
            del_map.append(pos)
            self.valid[tuple(pos)] = 0
                    
        return np.array(del_map), self.valid


    def walk_dis(self, input, dis = 0, pre = []):
        """ Recursive function to walk to all sites
            within a distance of min_dis jumps """
        if self.min_dis == 0:
            return input
        
        for i, elem in enumerate(input):
            if isinstance(elem, (np.ndarray, np.generic)):
                input[i] = elem.tolist()


        neigh = []
        for pos in input:
            suggest, _ = self.connected_neigh(pos)
            for s in suggest:
                s_in_pre = np.any(np.all(s == pre, axis = -1))
                s_in_neigh = np.any(np.all(s == neigh, axis = -1))
                if not s_in_pre and not s_in_neigh:
                        neigh.append(s)
            
        dis += 1
        if dis >= self.min_dis:
            return input + neigh
        else:
            pre = np.array(input)
            return  np.concatenate((pre, self.walk_dis(neigh, dis, pre)))
          
    
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
        # Partition
        L = int(np.ceil(np.sqrt(self.num_walks)))
        size = np.shape(self.mat)
     
        grid = []
        for i in range(L):
            xstart = (i*size[0])//L
            xstop = ((i+1)*size[0])//L 
            xpoint = xstart + (xstop-xstart)//2
            for j in range(L):
                ystart = (j*size[1])//L
                ystop = ((j+1)*size[1])//L 
                ypoint = ystart + (ystop-ystart)//2
                grid.append([xpoint, ypoint])
                
        grid = np.array(grid)
        
        # Ordering for even distributed points
        if self.num_walks == len(grid):
            pass # No need to order
        elif L > 1:
            order = []
            # Start at lower left quarter center
            start =  np.array([size[0]//4, size[1]//4])
            start_diff = np.linalg.norm(grid-start, axis = 1)
            order.append(np.argmin(start_diff))
            
            left = np.arange(len(grid))
            while len(order) < self.num_walks:
                dis = np.linalg.norm(grid[order, np.newaxis]-grid[left], axis = 2)
                dis_min = np.min(dis, axis = 0)
                idx = np.random.choice(np.where(np.isclose(dis_min, dis_min.max()))[0])
                order.append(idx)
            grid = grid[order]
        
        idx = grid[self.valid[grid[:, 0], grid[:,1]] == 1]
        return idx
        
        
        
        

if __name__ == "__main__":
    
    RW = RW_Generator()
    

    
    # mat = RN.generate()
    
    
    