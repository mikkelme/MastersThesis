from ase.build import graphene_nanoribbon
from ase.build import graphene

from ase.io import  lammpsdata
from ase.visualize import view
import numpy as np




def build_graphene_sheet(mat, view_lattice = False, write = False):
    Cdis = 1.42 # carbon-carbon distance [Å]

    
    shape_error = f"SHAPE ERROR: Got matrix of shape {np.shape(mat)}, y-axis must be mut multiple of 2 and both nonzero integer."
    assert mat.shape[0]%1 == 0 and mat.shape[1]%1 == 0 and mat.shape[1]%2 == 0, shape_error
    assert mat.shape[0] != 0 and mat.shape[1] != 0, shape_error

    xlen = mat.shape[0]
    ylen = mat.shape[1]//2
  
    # --- Create graphene lattice --- #
    atoms = graphene_nanoribbon(xlen, ylen, type='zigzag', saturated=False, C_C=Cdis, vacuum=2.0)
    atoms.pbc = [False, False, False] # Set x,y,z to non periodic (not sure if this is relevant)


    # Swap axes: y <-> z
    new_posistions = atoms.get_positions()[:,(0,2,1)]
    new_cell = (atoms.get_cell()[(0,2,1), :])[:, (0,2,1)]
    eps = 1e-12


    # Readjust cell to fit atoms 
    new_posistions[:,1] += eps  # Move atoms in y-direction by eps (avoid cell boundart) 
    ymax = np.max(new_posistions[:,1]) # Find new ymax
    new_cell[1,1] = ymax + eps # Put cell ymax just over the last atom 


    atoms.set_positions(new_posistions)
    atoms.set_cell(new_cell)

    #--- Reorder atoms ---#
    yline_len = 2*ylen 
    for i in range(1,xlen, 2):
        for j in range(yline_len):
            from_idx = i*yline_len + j
            to_idx = (i+1)*yline_len-1
            atoms.positions[[from_idx, to_idx]] = atoms.positions[[to_idx, from_idx]]


    # --- Delete atoms --- #
    for i in reversed(range(mat.shape[0])):
        for j in reversed(range(mat.shape[1])):
            if mat[i,j] == 0:
                del atoms[i*yline_len+j]

   


    
    if view_lattice: 
        view(atoms)

    if write:
        lammpsdata.write_lammps_data('./lammps_sheet', atoms)

    return atoms


    return


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

            up = current_elem[0]%2 == 0
            sign = 1-2*up
            diff = next_elem - current_elem            
            absdiff = abs(diff)

            correction = absdiff[0] - (absdiff[0]==absdiff[1])
            direction = [diff[0], diff[1] + sign * correction]
            local_pair = mapping[direction[0], direction[1]] # local cordinates to center elem

            neigh = center_neigh(current_elem)
            global_pair = neigh[local_pair[:, 0], local_pair[:, 1]] # global atoms coordinates 

     
            [delete.append(pair) for pair in global_pair]



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


def delete_atoms(mat, delete_map):
    """ Remove valid atoms from atom matrix """ 
    m, n = np.shape(mat)   
    condition = np.logical_and(delete_map < (m,n), delete_map >= (0,0))
    delete_map = delete_map[np.all(condition, axis = 1), :]

    if len(delete_map > 0):
        mat[delete_map[:, 0], delete_map[:, 1]] = 0
    return mat




def pop_up_pattern(multiples, unitsize = (5,7), sp = 1, view_lattice = False):

    # --- Parameters --- #
    mat = np.ones((multiples[0]*10, multiples[1]*10)) # lattice matrix
    ref = np.array([0, 0]) # reference center element
    size = unitsize # Size of pop_up pattern

    assert (np.abs(size[0] - size[1]) - 2)%4 == 0, f"Unit size = {size} did not fulfill: |size[1]-size[0]| = 2, 4, 6, 10..."
    assert np.min(size) > 0, f"Unit size: {size} must have positives entries."
   
    # --- Set up cut out pattern --- #
    # Define axis for pattern cut out
    m, n = np.shape(mat)
    axis1 = np.array([2*(2 + sp + size[0]//2), 2 + sp + size[0]//2]) # up right
    axis2 = np.array([- 2*(1 + size[1]//2 + sp), 3*(1 + size[1]//2 + sp)]) # up left
    unit2_axis =  np.array([3 + size[0]//2 + size[1]//2, 1 + size[0]//4 + size[1]//4 - size[1]]) + (2*sp, -sp) # 2nd unit relative to ref

    ########## Testing indexes ########## 
    # inputs = [(3,1), (7,1), (11,1), (1,3), (5,3), (3, 5), (1, 7), (5, 7), (1, 11), (5, 11), (9,11), (1,13), (3,13), (5,13), (7,13), (9,13), (11,13)]
    # expected  = [[6, -1], [8,0], [10,1], [6, -3], [8,-2], [8, -4], [8, -6], [10, -5], [10, -9], [12, -8], [14, -7],[11,-10], [12,-10], [13,-9], [14,-9], [15,-8], [16,-8]]

    # for i, size in enumerate(inputs):
    #     unit2_axis =  np.array([5 + size[0]//2 + size[1]//2,  size[0]//4 - size[1] + size[1]//4]) # 2nd unit relative to ref
    #     check = "X"
        
    #     if np.all(unit2_axis == expected[i]):
    #         check = "√"
    #     print(f"size = {size} => {unit2_axis}, must be {expected[i]} ({check})" )
 
    # print("axis1", axis1)
    # print("axis2", axis2)
    # exit()
    ########## ########## ########## 

 
    # Create unit1 and unit2
    up = ref[0]%2 == 0
    line1 = [ref]
    line2 = []
    
    if up:
        for i in range((size[0]-1)//2):
            line1.append(ref - [i+1, (i+1)//2 ])
            line1.append(ref + [i + 1, i//2 + 1])

        for i in range(sp, size[1] + sp):
            line2.append(ref + [i+1, -(i + (i+1)//2 + 1)])

       
    else:
        for i in range((size[0]-1)//2):
            line1.append(ref + [i+1, (i+1)//2 ])
            line1.append(ref - [i + 1, i//2 + 1])


        for i in range(sp, size[1] + sp):
            line2.append(ref + [i+1, -(i + i//2 + 2)]) 

    
    del_unit1 = np.array(line1 + line2)
    del_unit2 = np.array(line1 + line2) + unit2_axis


    # --- Translate cut-out-units across lattice --- # 
    # Estimate how far to translate
    range1 = int(np.ceil(np.dot(np.array([m,n]), axis1)/np.dot(axis1, axis1))) + 1      # project top-right corner on axis 1 vector
    range2 = int(np.ceil(np.dot(np.array([0,n]), axis2)/np.dot(axis2, axis2)/2))  + 1   # project top-left corner on axis 2 vector


    # Translate and cut out
    for i in range(range1):
        for j in range(-range2, range2+1):
            vec = i*axis1 + j*axis2 
            del_map1 = del_unit1 + vec
            del_map2 = del_unit2 + vec

            mat = delete_atoms(mat, center_elem_trans_to_atoms(del_map1, full = True))
            mat = delete_atoms(mat, center_elem_trans_to_atoms(del_map2, full = True))


    # Build sheet from final matrix
    build_graphene_sheet(mat, view_lattice=view_lattice)
    return mat



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




if __name__ == "__main__":
    multiples = (4, 8)
    unitsize = (3, 5)
    mat = pop_up_pattern(multiples, unitsize, sp = 2, view_lattice = True)


    # multiples = (6, 12)
    # unitsize = (9,11)
    # mat = pop_up_pattern(multiples, unitsize,  view_lattice = True)
    # build_pull_blocks(mat, sideblock = 0)
    # build_graphene_sheet(mat, view_lattice = False, write=True)
    # exit()

    # exit()
    # mat = np.ones((10, 10)) # Why does (5, 12) not work?
    # trans = np.array([[2,0], [3,1], [3,2], [3,3], [3,4], [4,3], [5,4]])
    # mat[:,0] = 0
    # trans = np.array([[20,0]])
    # exit()
    # delete_map = center_elem_trans_to_atoms(trans, full = True)   
    # mat = delete_atoms(mat, delete_map)
   

   
    # build_graphene_sheet(mat, view_lattice = True, write = False)

   




