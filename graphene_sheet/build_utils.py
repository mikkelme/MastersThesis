import numpy as np

def delete_atoms(mat, delete_map):
    """ Remove valid atoms from atom matrix """ 
    m, n = np.shape(mat)   
    condition = np.logical_and(delete_map < (m,n), delete_map >= (0,0))
    delete_map = delete_map[np.all(condition, axis = 1), :]

    if len(delete_map > 0):
        mat[delete_map[:, 0], delete_map[:, 1]] = 0
    return mat



def center_elem_trans_to_atoms(trans, full = False):
    """ Gather atom pairs for deletion when crossing to a new center element """
    """ full = Delete all neighbours """
    mapping = np.zeros((3,3, 2, 2), dtype = int)
    mapping[0,1] = [0,2], [1,2]       # up
    mapping[1,1] = [1,2], [1,1]       # right-up
    mapping[1,-1] = [1,1], [1,0]      # right-down
    mapping[0,-1] = [1,0], [0,0]     # down
    mapping[-1,-1] = [0,0], [0,1]   # left-down
    mapping[-1,1] = [0,1], [0,2]    # up-left


    delete = []
    num_trans = len(trans) - 1
    if not full:
        for i in range(num_trans):
            current_elem = trans[i]
            next_elem = trans[i+1]
            if (current_elem == next_elem).all(): continue

            up = current_elem[0]%2 == 0
            sign = 1-2*up
            diff = next_elem - current_elem            
            absdiff = abs(diff)

            correction = absdiff[0] - (absdiff[0]==absdiff[1])
            direction = [diff[0], diff[1] + sign * correction]


            # Error to deal with more formula later on...
            if np.abs(current_elem[0]-next_elem[0]) == 2:
                print("ERROR", current_elem, next_elem, direction)    
                exit()
            if np.abs(current_elem[1]-next_elem[1]) == 2:
                print("ERROR", current_elem, next_elem, direction)    
                exit()


            try:
                local_pair = mapping[direction[0], direction[1]] # local cordinates to center elem
            except IndexError:
                print(f"Center element trans error: Center elements {current_elem} and {next_elem} are not neighbours.")
                exit(1)



            neigh = center_neigh(current_elem)
            global_pair = neigh[local_pair[:, 0], local_pair[:, 1]] # global atoms coordinates 

            [delete.append(pair) for pair in global_pair]
            # if i == 1:  break


    else:
        for i in range(num_trans + 1):
            current_elem = trans[i]
            global_atoms = center_neigh(current_elem).astype("int").reshape(6,2)
            [delete.append(atom) for atom in global_atoms]
          


    return np.array(delete, dtype = int)



def center_neigh(center_elem):
    """ Return all neighbour atoms to a center element """
    neigh = np.zeros((2,3,2))
    n, m = center_elem

    m_start = 2*m -n%2
    m_end = m_start + 2 
    for i in range(2):
        for j in range(3):
            neigh[i, j] = [n-1 + i, m_start + j]
    

    return neigh




def build_pull_blocks(mat, pullblock = 6, sideblock = 0):
    """ Add blocks on the x-z plane on the +-y sides  """
    m, n = np.shape(mat)

    # Try adding sideblocks
    if sideblock > 0:
        new_mat = np.ones((m+2*sideblock,n + 2*pullblock))
        new_mat[sideblock:-sideblock,pullblock:-pullblock] = mat
    else:
        new_mat = np.ones((m,n + 2*pullblock))
        new_mat[:,pullblock:-pullblock] = mat
    
    return new_mat, pullblock
