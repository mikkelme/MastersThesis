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
                        bias = [(0, 0), 0],
                        center_elem = True,
                        avoid_unvalid = False,
                        RN6 = False,
                        grid_start = True,
                        centering = False,
                        stay_or_break = 0,
                        avoid_clustering = 10,
                        periodic = True,
                        seed = None):


        # Check shape
        shape_error = f"SHAPE ERROR: Got size {size}, y-axis must be multiple of 2 and both nonzero positive integers."
        assert size[0]%1 == 0 and size[1]%1 == 0 and size[1]%2 == 0, shape_error
        assert size[0] > 0 and size[1] > 0, shape_error
            
    
        # if stay_or_break > 0:
        #     assert center_elem is not False, f"center_elem = {center_elem} is not valid in mode stay_or_break = {stay_or_break} > 0"
    
        # Convert variables
        self.size = np.array(size)
        self.num_walks = num_walks
        self.max_steps = max_steps
        self.min_dis = min_dis
        self.bias = bias
        self.center_elem = center_elem
        self.avoid_unvalid = avoid_unvalid
        self.RN6 = RN6
        self.grid_start = grid_start
        self.centering = centering # Move CM as close to starting point as possible
        self.stay_or_break = stay_or_break
        self.avoid_clustering = avoid_clustering
        self.periodic = periodic
        self.seed = seed
        if seed is not None: # Not working properly... XXX
            np.random.seed(seed)
            
        # Possible directions
        self.main_center = connected_neigh_center_elem((0,0))[1] # 6 main directions based on the center elements
        
        # Get equivalent atom direction when following center direction
        main_atom_even = connected_neigh_atom((0,0))[1]
        main_atom_odd = connected_neigh_atom((1,0))[1]
        
        # Distinguish between odd or even positions
        # (idx[0]+idx[1])%2 => 0: even, 1: odd
        self.center_to_atom_even = [] 
        self.center_to_atom_odd = []
        for i, main in enumerate(self.main_center):
            best_even_idx = np.argmin(np.linalg.norm(main - main_atom_even, axis = 1))
            best_odd_idx = np.argmin(np.linalg.norm(main - main_atom_odd, axis = 1))
            
            self.center_to_atom_even.append(main_atom_even[best_even_idx])
            self.center_to_atom_odd.append(main_atom_odd[best_odd_idx])
            
        self.center_to_atom_even = np.array(self.center_to_atom_even)
        self.center_to_atom_odd = np.array(self.center_to_atom_odd)
    
    
    def initialize(self):
        """ Initialize matrices for walks and link to and setup
            correct neighbour connection  """
            
        # List for storing of delete maps
        self.total_del_map = []
            
        # Walk on center elements
        if self.center_elem is not False:
            self.working_size = np.array((int(self.size[0]), int(self.size[1]//2)))
            self.valid = np.ones(self.working_size, dtype = int)  # valid positions
            self.connected_neigh = connected_neigh_center_elem   # Neighbour connectivity  
        
        # Walk on atoms
        else:
            self.working_size = self.size
            self.valid = np.ones(self.size, dtype = int)  # valid positions
            self.connected_neigh = connected_neigh_atom   # Neighbour connectivity  
    
        # Check if periodicity is availble if needed
        if self.periodic:
            assert np.all(self.size%2 == 0), f"The size of the sheet {self.size} must have even side lengths to enable periodic boundaries."
    
        
        # Ideas for further features
        # TODO: Single walk copied to multiple locations
        # TODO: Distributions on RN walk length?
        # TODO: Consider using the proper random generator suggested by numpy.
        
   
    def generate(self): 
        """ Generate random walk pattern """
        self.initialize()
        grid = self.get_grid()
           
        # --- Individual RN walks --- #
        for w in range(self.num_walks):
            # Grid start
            if self.grid_start: # Find valid option on grid (already in RN order)
                idx = grid[self.valid[grid[:, 0], grid[:,1]] == 1]
                if len(idx) == 0: break
                start = idx[0]
                grid = grid[1:] # Remove option from grid
            else: #  Random valid start choice
                idx = np.argwhere(self.valid == 1)
                if len(idx) == 0: break
                start = random.choice(idx) 
                
            # Centering
            if self.centering: # Walk and move CM towards starting point
                prev_valid = self.valid.copy()
                del_map = self.walk(start)
                del_map = self.center_walk(del_map, prev_valid)
            else: # Walk and leave it
                del_map = self.walk(start)
            
            # Store del_map
            self.total_del_map.append(del_map) 
        
            # Make nearby sites unvalid 
            self.add_dis_bound(del_map) 
      

        # Concatenate walkers in delete map
        del_map = np.concatenate(self.total_del_map)
        
        # --- Transform from center elements to atoms --- #
        if self.center_elem is not False: 
    
            # Remove only intersecting atoms in center jumps
            if self.center_elem == 'intersect': 
                tmp_del_map = []
                for local_del_map in self.total_del_map:
                    if self.periodic:
                        local_del_map = self.unravel_PB(local_del_map, self.working_size)
                        local_del_map = center_elem_trans_to_atoms(local_del_map, full = False) 
                    else:
                        local_del_map = center_elem_trans_to_atoms(del_map, full = False) 
                
                    if len(local_del_map) == 0: continue
                    tmp_del_map.append(local_del_map)
                    
                del_map = np.concatenate(tmp_del_map)
            
            # Remove all atoms surrounding center elements
            elif self.center_elem == 'full':
                del_map = center_elem_trans_to_atoms(del_map, full = True) 
        
            else:
                exit(f"center_elem = {self.center_elem} not understood")
                    
            
            if self.periodic and len(del_map) > 0:
                del_map = self.PB(del_map, self.size)
                
            
        # Initialize configuration matrix and apply cuts
        self.mat = np.ones(self.size, dtype = int) 
        self.mat = delete_atoms(self.mat, del_map)
            
                
        # --- Avoid isolated clusters --- #
        if self.avoid_clustering is not False:
            self.vertical_spanning = None
            
            # Create binary matrix for visited sites (1 = visited)
            self.visit = np.zeros(self.size, dtype = int)
            
            # First bottom left site available
            x_start = (np.argwhere(self.mat[:, 0] == 1)).ravel()
            if len(x_start) == 0: # Cannot span vertically
                self.vertical_spanning = False
            else:
                start = (x_start[0], 0)
                
                
    
            # Walk from all bottom sites and check if vertical spanning
            # and then walk from all top sites
            while self.vertical_spanning is not False: 
                self.DFS(start, PB = False)
                
                
                bottom_x = np.argwhere (self.mat[:, 0] - self.visit[:, 0] > 0).ravel()
                top_x = np.argwhere (self.mat[:, -1] - self.visit[:, -1] > 0).ravel()
                if len(bottom_x) > 0: # Walk from bottom
                    start = [bottom_x[0], 0]
                elif self.vertical_spanning is None: # Check if vertical spanning
                    top_reached = np.sum(self.visit[:, -1]) > 0
                    self.vertical_spanning = top_reached
                elif len(top_x) > 0: # Walk from top
                    start = (top_x[0], self.size[1]-1)
                else:
                    break
                    
            # Check if all sites are visited (is multiple clusters present)
            all_visited = np.sum(self.mat - self.visit) > 0.5
            not_vertically_spanning = self.vertical_spanning != True
            detect = all_visited or not_vertically_spanning
                        
            if detect: # Isolated cluster detected
                self.avoid_clustering -= 1
                print(f'Isolated cluster detected | {self.avoid_clustering} attempts left')
                if self.avoid_clustering > 0:
                    self.generate()
                else:
                    
                    if not_vertically_spanning:
                        print("Not possible to get vertical spanning cluster")
                        self.mat = None
                    else:
                        print('Removing non-spanning clusters')
                        self.mat = self.visit.copy() 
                    
            # Returns multiple times, but all with the correct matrix though...
            # Not sure if this is a problem XXX
        return self.mat
            
   
    def walk(self, start):
        """ Perform a single random walk""" 
        self.valid[tuple(start)] = 0 # Mark starting point as unvalid
        del_map = [start] # Iniziate delete map


        if self.stay_or_break > 0:
            self.last_direction = None # rounded to nearest of the six
        
        # Random direction among 6 principal graphene directions (center elem)
        if self.RN6:
            six_directions = connected_neigh_center_elem((0,0))[1]
            bias_dir = six_directions[np.random.choice(len(six_directions), 1)[0]]
            self.bias[0] = bias_dir
            
            if self.stay_or_break > 0:
                self.last_direction = bias_dir


    
        pos = start    
        for i in range(self.max_steps): # Walk loop
            # Get neighbours and directions
            neigh, direction = self.connected_neigh(pos)
            
            if self.periodic:
                neigh = self.PB(neigh, self.working_size)
            
            # Check which neighbours is avaliable (on sheet and noncut)
            on_sheet = np.all(np.logical_and(neigh < self.working_size, neigh >= (0,0)), axis = 1)
            idx = np.argwhere(on_sheet)[:,0]
            available = on_sheet
            available[idx] = self.valid[neigh[on_sheet][:,0], neigh[on_sheet][:,1]] == 1
          
           
            # Manage unavailable sites
            if self.avoid_unvalid: # Remove all unavailble 
                neigh = neigh[available]
                direction = direction[available]
                available = available[available]
                if len(neigh) == 0: break # No where to go
            else: # Remove neighbours already visited in this walk
                map = []
                for i in range(len(neigh)):
                    map.append(~np.any(np.all(neigh[i] == del_map, axis = 1)))
                neigh = neigh[map]
                direction = direction[map]
                available = available[map]
                
            if len(neigh) == 0: break # No where to go
                
            # Choose next site taking bias into account
            p = self.get_p(direction)
            
          
          
            if self.stay_or_break > 0:  # Stay on direction (if possible) by prob stay_or_break
            
                # Whether the element is considered odd or even              
                if self.center_elem == False:
                    odd_even = (pos[0]+pos[1])%2 # even = 0, odd = 1
                    if odd_even == 0: # even     
                        center_to_atom = self.center_to_atom_even
                    else: # odd_even == 1 # odd
                        center_to_atom = self.center_to_atom_odd
            
                if self.last_direction is not None:
                    
                    # Calculate distance between last direction and possible directions 
                    if self.center_elem == False:
                        dis = np.linalg.norm(self.main_center - self.last_direction, axis = 1)
                    else:
                        dis = np.linalg.norm(direction - self.last_direction, axis = 1)
                    
                    
                    if np.any(dis < 1e-3): # Direction can be maintained
                        stay = np.random.random() < self.stay_or_break
                        mask = dis < 1e-3
                                                    
                        
                        if stay: # Stay on direction
                            if self.center_elem == False:
                                target_direction = center_to_atom[mask].ravel()                 
                            else:
                                target_direction = self.last_direction
                            
                            choice = np.argmin(np.linalg.norm(target_direction - direction, axis = 1))

                        else: # Change direction
                            if self.center_elem == False:
                                
                                dis_matrix = np.array([np.linalg.norm(center_to_atom - d, axis= 1) for d in direction])
                                mapping = np.argwhere(dis_matrix < 1e-3)
                                center_p = np.zeros(len(self.main_center))
                            
                            
                                for (i,j) in mapping:
                                    center_p[j] = p[i]
                              
                              
                                center_p[mask] = 0 # Already ruled this out
                                center_p[~mask] /= np.sum(center_p[~mask])
                                
                              
                                center_choice = np.random.choice(len(self.main_center), p = center_p)
                                self.last_direction = self.main_center[center_choice]
                                
                                atom_direction = center_to_atom[center_choice]
                                choice = np.argmin(np.linalg.norm(atom_direction - direction, axis = 1))
                            
                            else:
                                p[mask] = 0 # Already ruled this out
                                p[~mask] /= np.sum(p[~mask])
                              
                              
                                if np.sum(p) == 0: # No where to go
                                    break
                                    
                                p /= np.sum(p) # Normalize again for safety measures
                                choice = np.random.choice(len(neigh), p = p)
                                self.last_direction = direction[choice]
                                 
                    else: # Direction cannot be maintained
                        self.last_direction = None
                
                
                if self.last_direction is None: # Last direction is not yet set or direction cannot be maintained causes self.last_direction = None
                        
                        choice = np.random.choice(len(neigh), p = p)
                        
                        if self.center_elem == False:
                            center_cand = np.argwhere(np.linalg.norm(center_to_atom - direction[choice], axis = 1) < 1e-3).ravel()
                            p = self.get_p(self.main_center[center_cand])
                            center_choice = center_cand[np.random.choice(len(center_cand), p = p)]
                            self.last_direction = self.main_center[center_choice]
                            
                            atom_direction = center_to_atom[center_choice]
                            choice = np.argmin(np.linalg.norm(atom_direction - direction, axis = 1))
                            
                        else:
                            self.last_direction = direction[choice]
                            
                            
            else: # OFF: self.stay_or_break = 0
                choice = np.random.choice(len(neigh), p = p)

             
            if available[choice] == False: # Hit unvalid site
                break
            
            # Update position, delete map and valid matrix
            pos = neigh[choice] 
            del_map.append(pos)
            self.valid[tuple(pos)] = 0
        return np.array(del_map)


    def add_dis_bound(self, walk):
        """ Mark sites within min_dis walking distance as unvalid """
        for w in walk:
            new_del_map = np.array(self.walk_dis([w]))
            if self.periodic:
                new_del_map = self.PB(new_del_map, self.size)
            self.valid = delete_atoms(self.valid, new_del_map)
    

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
    
    
    def center_walk(self, del_map, prev_valid):  
        """ Move center of mass (CM) as close to the starting point as possible """

        # Unravel PB discontinuous jumps
        if self.periodic:
            del_map = self.unravel_PB(del_map, self.working_size)
            
    
        # Get start and approximate CM
        start = del_map[0] # Starting point
        CM = np.round(np.mean(del_map, axis = 0)) # Approximate CM
        
        # Define path to relocate RM to start
        continuous_path = np.linspace(0, start-CM, 2*int(np.linalg.norm(start - CM)))
        if self.center_elem is not False: # jumps of even x
            discrete_path = np.unique(np.round(continuous_path), axis = 0).astype(int)
            mask = discrete_path[:,0]%2 == 0 # Even x-jumps
            discrete_path = discrete_path[mask]
            
        else: # Jumps of even x and y
            discrete_path = np.unique(np.round(continuous_path/2)*2, axis = 0).astype(int) 
    

        # Make sure that path is ordered to end at (0,0)
        idx = np.flip(np.argsort(np.sum(np.abs(discrete_path), axis = 1)))
        discrete_path = discrete_path[idx]
        
        # Move walk CM to start and backtrack
        # until valid position is found
        try_map = del_map
        for trans in discrete_path:
            try_map = del_map + trans
            
            if self.periodic:
                try_map = self.PB(try_map, self.working_size)
            else:
                on_sheet = np.all(np.logical_and(try_map < self.working_size, try_map >= (0,0)), axis = 1)
                if not np.all(on_sheet):
                    continue
          
            valid = np.all(prev_valid[try_map[:,0], try_map[:,1]])
            if valid:
                break
        
        self.valid = prev_valid.copy() # New unvalid sites are added in add_dis_bond()
        return try_map
      

    
    def DFS(self, pos, PB = False):
        """ Depth-first search (DFS) used for 
            detecting isolated clusters (walking on atoms not centers) """


        # Check is visited
        if self.visit[pos[0], pos[1]] == 1:
            return # Already visited
      
        # Mark as visited
        self.visit[pos[0], pos[1]] = 1
            
        # Find potential neighbours (with PB)
        neigh, _ = connected_neigh_atom(pos)
        if PB:
            neigh = self.PB(neigh, self.size)
        else:
            on_sheet = np.all(np.logical_and(neigh < self.size, neigh >= (0,0)), axis = 1)
            neigh = neigh[on_sheet]
            
        # Start new search if neighbour atoms is present
        for pos in neigh:
            if self.mat[pos[0], pos[1]] == 1: # Atom is present
                self.DFS(pos, PB)
                
                
    def PB(self, array, size):
        """ Apply periodic boundary conditions """
        return (array + size)%size
    
    
    def unravel_PB(self, del_map, size):
        """ Unravel periodic boundary conditions """
        diff = np.abs(del_map[1:] - del_map[:-1])
        for d in np.argwhere(diff > 2): # d = [num_cut, axis]
            sign = -np.sign(del_map[d[0]+1, d[1]] - del_map[d[0], d[1]])
            del_map[d[0]+1:, d[1]] += sign*size[d[1]]  
        
        return del_map
    
    
    
    def get_p(self, input_dir):
        """ Get discrete transistion probabilities for valid sites provided by input_dir
            by using a Gibbs–Boltzmann distribution """
        
        bias_dir, bias_strength = self.bias
    
        if bias_strength == 0: # Non biased
            return np.ones(len(input_dir))/len(input_dir)

        # Check that bias direction properly defined
        assert np.linalg.norm(bias_dir) > 0, f"force direction {bias_dir} has zero norm"

        # --- Gibbs–Boltzmann distribution --- #
        # Energy difference is calculated assuming a unit jump in direction of input_dir
        # under influence of a force in bias_dir direction and |F|/kT = bias_strength

        DeltaE = -bias_strength*np.dot(input_dir, bias_dir)/(np.linalg.norm(bias_dir)*np.linalg.norm(input_dir, axis = 1))
        expE = np.exp(-DeltaE)
        p = expE/np.sum(expE)
        return p
    
    

    def get_grid(self):
        """ Create a square grid for starting points """
        # Partition into smallest L*L sites allowing space for all walkers
        L = int(np.ceil(np.sqrt(self.num_walks)))
      
        # Define grid
        grid = []
        for i in range(L):
            xstart = (i * self.working_size[0]) // L
            xstop  = ((i+1) * self.working_size[0]) // L 
            xpoint = xstart + (xstop-xstart)//2
            for j in range(L):
                ystart = (j * self.working_size[1]) // L
                ystop  = ((j+1) * self.working_size[1]) // L 
                ypoint = ystart + (ystop-ystart)//2
                grid.append([xpoint, ypoint])       
        grid = np.array(grid)
        
        # Order grid to achieve evenly distributed points
        order = []
        
        # Start at lower left quarter center
        start =  np.array([self.working_size[0]//4, self.working_size[1]//4])
        start_diff = np.linalg.norm(grid-start, axis = 1)
        order.append(np.argmin(start_diff))
        
        # Choose remaining points to maximize distance 
        # to already placed points
        left = np.arange(len(grid))
        while len(order) < self.num_walks:
            dis = np.linalg.norm(grid[order, np.newaxis] - grid[left], axis = 2)
            dis_min = np.min(dis, axis = 0)
            idx = np.random.choice(np.where(np.isclose(dis_min, dis_min.max()))[0])
            order.append(idx)
        grid = grid[order]
    
        return grid
        
    def __str__(self):
        s = f'size = {self.size}, \
num_walks = {self.num_walks}, \
max_steps = {self.max_steps}, \
min_dis = {self.min_dis}, \
bias = {self.bias}, \
center_elem = {self.center_elem}, \
avoid_unvalid = {self.avoid_unvalid}, \
RN6 = {self.RN6}, \
grid_start = {self.grid_start}, \
centering = {self.centering}, \
stay_or_break{self.stay_or_break}, \
avoid_clustering = {self.avoid_clustering}, \
periodic = {self.periodic}, \
seed = {self.seed}'
        return s

if __name__ == "__main__":
    
    RW = RW_Generator()
    print(RW)

    
    # mat = RN.generate()
    
    
    