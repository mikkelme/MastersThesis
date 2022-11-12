import sys
sys.path.append('../') # parent folder: MastersThesis

from graphene_sheet.build_utils import *
import random

def walk(start, valid, max_steps):
    valid[tuple(start)] = 0
    
    pos = start
    del_map = []
    for i in range(max_steps):
        neigh = connected_neigh(valid, pos)
        if len(neigh) == 0: # No where to go
            break
        choice = random.randint(0, len(neigh)-1)
        pos = neigh[choice] 
        
        del_map.append(pos)
        valid[tuple(pos)] = 0.0
        
    
    return np.array(del_map), valid

def RN(size = (50, 70), num_walks = 1, max_steps = 1, max_dis = 10, uniform = True):
    
    
    size = (5, 10)
    mat = np.ones(size) # lattice matrix
    valid = mat.copy()  # valid positions
    
    if uniform:
        # TODO: place site starts uniformly 
        # How can one place N points with greatest distance in a box?
        # Without going heavy on math theory here...
        
        # Single walk case
        x = int(mat.shape[0]//(num_walks + 1))
        y = int(mat.shape[1]//(num_walks + 1)) 
        print(x,y)
        
        exit()
        
    for w in range(num_walks):
        idx = np.argwhere(valid == 1)
        if len(idx) == 0:
            print("no more atoms")
            break
        
        start = random.choice(idx)
        start = (x,y)
        del_map, valid = walk(start, valid, max_steps)
        mat = delete_atoms(mat, del_map)
        valid = add_dis_bound(del_map, valid, max_dis)



    # # invert 
    # mat[mat == 0] = -1
    # mat[mat == 1] = 0
    # mat[mat == -1] = 1
    
    # valid[valid == 0] = -1
    # valid[valid == 1] = 0
    # valid[valid == -1] = 1
    # # return valid
    
    
    return mat




if __name__ == "__main__":
    pass
    