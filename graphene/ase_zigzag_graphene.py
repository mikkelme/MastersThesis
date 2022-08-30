from ase.build import graphene_nanoribbon
from ase.build import graphene

from ase.io import  lammpsdata
from ase.visualize import view
import numpy as np




def build_graphene_sheet(mat, view_lattice = False):
    Cdis = 1.42 # carbon-carbon distance [Å]

    
    shape_error = f"SHAPE ERROR: Got matrix of shape {np.shape(mat)}, y-axis must be mut multiple of 2 and both nonzero integer."
    assert mat.shape[0]%1 == 0 and mat.shape[1]%1 == 0 and mat.shape[1]%2 == 0, shape_error
    assert mat.shape[0] != 0 and mat.shape[1] != 0, shape_error

    xlen = mat.shape[0]
    ylen = mat.shape[1]//2
    # xlen = 2
    # ylen = 2
  
    # --- Create graphene lattice --- #
    atoms = graphene_nanoribbon(xlen, ylen, type='zigzag', saturated=False, C_C=Cdis, vacuum=2.0)
    atoms.pbc = [False, False, False] # Set x,y,z to non periodic (not sure if this is relevant)


    # Swap axes: y <-> z
    new_posistions = atoms.get_positions()[:,(0,2,1)]
    new_cell = atoms.get_cell()[(0,2,1),(0,2,1)]
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
                # print(atoms[i*yline_len+j].index)
                del atoms[i*yline_len+j]

    
    if view_lattice: 
        view(atoms)

    lammpsdata.write_lammps_data('./lammps_sheet', atoms)



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




def pop_up_pattern():

    # Settings
    mat = np.ones((40, 80)) # lattice matrix
    ref = np.array([0, 0]) # reference center element

    size = (5,3) # Size of pop_up pattern
    # Note: Only odd values allowed and |size[1]-size[0]| = 2, 6, 10...
    # Not allowed: (1,1), (3, 3), (5,1), (5,5)...
    # Allowed: (1,3), (5,3), (3,1), (7,1)...


    # 
    # m, n = np.shape(mat)

    axis1 = np.array([6 + 2*(size[0]//2), 3 + size[0]//2]) # up right
    axis2 = np.array([-4 - 2*(size[1]//2), 6 + 3*(size[1]//2)]) # up left
    unit2_axis =  np.array([5 + size[0]//2 + size[1]//2, -2 + size[0]//3 - size[1]//2 - size[1]//5])


 
    up = ref[0]%2 == 0
    line1 = [ref]
    line2 = []
    if up:
        for i in range((size[0]-1)//2):
            line1.append(ref - [i+1, (i+1)//2 ])
            line1.append(ref + [i + 1, i//2 + 1])

        for i in range(size[1]):
            line2.append(ref + [i+2, -(i + i//2 + 3)])
       
    else:
        for i in range((size[0]-1)//2):
            line1.append(ref + [i+1, (i+1)//2 ])
            line1.append(ref - [i + 1, i//2 + 1])


        for i in range(size[1] ):
            line2.append(ref + [i+2, -(i + (i+1)//2 + 3)])
       
   

    del_unit1 = np.array(line1 + line2)
    del_unit2 = np.array(line1 + line2) + unit2_axis


    range1 = int(np.ceil(np.dot(np.array([m,n]), axis1)/np.dot(axis1, axis1)))      # project top-right corner on axis 1 vector
    range2 = int(np.ceil(np.dot(np.array([0,n]), axis2)/np.dot(axis2, axis2)/2))    # project top-left corner on axis 2 vector

    for i in range(range1):
        for j in range(-range2, range2+1):
            vec = i*axis1 + j*axis2 
            del_map1 = del_unit1 + vec
            del_map2 = del_unit2 + vec

            mat = delete_atoms(mat, center_elem_trans_to_atoms(del_map1, full = True))
            mat = delete_atoms(mat, center_elem_trans_to_atoms(del_map2, full = True))





    build_graphene_sheet(mat, view_lattice = True)



if __name__ == "__main__":

    pop_up_pattern()


    exit()  
    mat = np.ones((5, 10)) # Why does (5, 12) not work?
    trans = np.array([[2,0], [3,1], [3,2], [3,3], [3,4], [4,3], [5,4]])
    # trans = np.array([[20,0]])
    # exit()
    delete_map = center_elem_trans_to_atoms(trans, full = True)   
    mat = delete_atoms(mat, delete_map)
   

   
    build_graphene_sheet(mat, view_lattice = True)

   




