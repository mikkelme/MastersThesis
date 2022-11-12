import numpy as np

import matplotlib.pyplot as plt # TMP XXX

def delete_atoms(mat, delete_map):
    """ Remove valid atoms from atom matrix
    
    :param mat: configuration matrix
    :type mat: numpy.ndarray(), shape (x, y)
    
    :param delete_map: list of atoms to delete
    :type delelta_map: numpy.ndarray(), shape = (atoms, 2)
    
    """

    if len(delete_map) == 0:
        return mat

    m, n = np.shape(mat)   
    condition = np.all(np.logical_and(delete_map < (m,n), delete_map >= (0,0)), axis = 1)
    delete_map = delete_map[condition, :]
    

    if len(delete_map > 0):
        mat[delete_map[:, 0], delete_map[:, 1]] = 0
    return mat



def center_elem_trans_to_atoms(trans, full = False):
    """ Gather atom pairs for deletion when crossing to a new center element """
    """ full = Delete all neighbours """
    mapping = np.zeros((3,3, 2, 2), dtype = int)
    mapping[0,1] = [0,2], [1,2]     # up
    mapping[1,1] = [1,2], [1,1]     # right-up
    mapping[1,-1] = [1,1], [1,0]    # right-down
    mapping[0,-1] = [1,0], [0,0]    # down
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


# TODO: mat -> valid, for consistensy 
def connected_neigh(valid, pos):
    """ Get three connected neightbours in sheet
        if they are valid (inside the sheet) """
    x, y = pos
    m, n = np.shape(valid)   
    
    Cdis = 1.42
    a = 3*Cdis/np.sqrt(3)
    x_ver = a*np.sqrt(3)/6  # vertical
    y_ver = a/2             # vertical
    
    neigh = np.array([[x, y+1], [x, y-1], [m,n]])
    if (x + y)%2: # Right
        neigh[2] = [x+1, y]   
        direction = np.array([[-x_ver, y_ver], [-x_ver, -y_ver], [Cdis, 0]])
    else: # Left
        neigh[2] = [x-1, y]  
        direction = np.array([[x_ver, y_ver], [x_ver, -y_ver], [-Cdis, 0]])
        
    
    # Check if atom is on sheet and non deleted
    on_sheet = np.all(np.logical_and(neigh < (m,n), neigh >= (0,0)), axis = 1)
    idx = np.argwhere(on_sheet)[:,0]
    available = on_sheet
    # tuplle instead of [0], [1] XXX
    available[idx] = valid[neigh[on_sheet][:,0], neigh[on_sheet][:,1]] == 1
    return neigh[available], direction[available]
    
  
def get_neighbour(pos):  
    x, y = pos
    
    neigh = [[x, y+1], [x, y-1]]
    if (x + y)%2: # Right
        neigh.append([x+1, y])   
    else: # Left
        neigh.append([x-1, y])   
        
    return neigh
    
def walk_dis(input, max_dis, dis = 0, pre = []):
    """ Recursive function to walk to all sites
        within a distance of max_dis jumps """
        
    if max_dis == 0:
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
    if dis >= max_dis:
        return input + neigh
    else:
        pre = input
        return pre + walk_dis(neigh, max_dis, dis, pre)
    

    
def add_dis_bound(walk, valid, max_dis):
    for w in walk:
        del_map = np.array(walk_dis([w], max_dis))
        valid = delete_atoms(valid, del_map)
    return valid


# def half_norm(x, mu, sigma):
#     """ shift so it integrates to 1 in x interval """
#     y = 2 * 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1/2*((x-mu)/sigma)**2)  
#     corr = np.trapz(2 * 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1/2*((x-mu)/sigma)**2), x)
#     return y/corr
    
    
def half_norm(x, mu, sigma):
    """ shift so it integrates to 1 in x interval """
    x_interval = np.linspace(0, 2*np.pi, int(1e4))
    y = 2 * 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1/2*((x-mu)/sigma)**2)
    shift = np.trapz(2 * 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1/2*((x_interval-mu)/sigma)**2), x_interval)
    return y/shift
    
        
def MATCH(input_dir, proj_dir, strength):
    if strength == 0:
        return np.ones(len(input_dir))/len(input_dir)
    
    
    norm = np.linalg.norm(input_dir, axis = 1)*np.linalg.norm(proj_dir)
    angle = np.arccos(np.dot(input_dir, proj_dir)/norm)
    

    # 0 => 0.1
    # 0.5 =>
    s = 1
    mu = 0
    sigma = 20*np.exp(-s*4)
    print(sigma)
    ## XXX 
    p = half_norm(angle, mu, sigma) / np.sum(half_norm(angle, mu, sigma))
    print(angle)
    print("p = ", p, "sum =", np.sum(p))
    
    
    exit()
    
    
    # ndist = lambda x: np.exp(-1/2*((x-mu)/sigma)**2)/(sigma * np.sqrt(2*np.pi)) # normal distribution
    
    print(corr)
    # xy_vec = np.vstack((x, y)).T

    # # Directions
    # dir_para = drag_direction.astype('float64')
    # dir_perp = np.array((dir_para[1], -dir_para[0]))

    # # Unit directions
    # dir_para /= np.linalg.norm(dir_para)
    # dir_perp /= np.linalg.norm(dir_perp)
    
    # # Projection
    # proj_para = np.dot(xy_vec, dir_para) 
    # proj_perp = np.dot(xy_vec, dir_perp)
    
    
  
    # return proj_para, proj_perp


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

