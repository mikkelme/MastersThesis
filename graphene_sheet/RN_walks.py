import sys
sys.path.append('../') # parent folder: MastersThesis

from graphene_sheet.build_utils import *
import random

def walk(start, mat, max_steps = 10):
    mat = delete_atoms(mat, np.array([start]))
    
    pos = start
    for i in range(max_steps):
        neigh = connected_neigh(mat, pos)
        if len(neigh) == 0: # No where to go
            break
        choice = random.randint(0, len(neigh)-1)
        pos = neigh[choice] 
        mat = delete_atoms(mat, np.array([pos]))
    
    return mat

def RN(size = (50, 70), num_walks = 20):
    
    # size = (5,10)
    mat = np.ones(size) # lattice matrix
    for w in range(num_walks):
        idx = np.argwhere(mat == 1)
        if len(idx) == 0:
            print("no more atoms")
            break
        
        start = random.choice(idx)
        mat = walk(start, mat)
        
    
    
    
        
    
    
    # pos = np.array([2, 4])
    # walk(pos, mat)
    
    # invert 
    # mat[mat == 0] = -1
    # mat[mat == 1] = 0
    # mat[mat == -1] = 1
    
    
    
    
    return mat





if __name__ == "__main__":
    RN()