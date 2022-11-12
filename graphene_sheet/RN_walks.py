import sys
sys.path.append('../') # parent folder: MastersThesis

from graphene_sheet.build_utils import *
import random

# TODO: change name: force <-> bias ?
def walk(start, valid, max_steps, force = [(0,1), 1]):
    valid[tuple(start)] = 0
    # TODO: Consider using the proper random generator
    #       suggested by numpy.
    
    pos = start
    del_map = []
    for i in range(max_steps):
        neigh, direction = connected_neigh(valid, pos)
        if len(neigh) == 0: # No where to go
            # XXX: Should the walker take the only valid option
            # or simply stop if it choose an uvalid option
            # among multiple possibilities?
            break
        
    
        force_direction = force[0]
        
        p = MATCH(direction, np.array(force[0]), force[1])
        print(p)
        # print(force_direction)
        # print(pos)
        # print(neigh)
        # print(direction)
        
        # choice = random.randint(0, len(neigh)-1)
        # test = np.random.choice(len(neigh), p = np.array([0.33, 0.33, 0.34]))
        # print(np.version.version)
        # print(np.random.choice(["a", "b", "c"], p=[0, 0, 1]))
        # print(neigh)
        # print("choose:", test)
        exit()
        pos = neigh[choice] 
        
        del_map.append(pos)
        valid[tuple(pos)] = 0.0
        
    
    return np.array(del_map), valid




    
    

def RN(size = (50, 70), num_walks = 1, max_steps = 15, max_dis = 1, uniform = False):
    
    
    size = (5, 10)
    mat = np.ones(size) # lattice matrix
    valid = mat.copy()  # valid positions
    
    walk([0,1], valid, max_steps)
    exit()
    
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
    