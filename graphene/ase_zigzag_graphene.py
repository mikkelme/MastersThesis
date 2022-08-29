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


def center_elem_trans_to_atoms(transistions, full = False):
    """ Gather atom pairs for deletion when crossing to a new center element """
    """ full = Delete all neighbours """
    mapping = np.zeros((3,3, 2, 2), dtype = int)
    mapping[0,1] = [0,2], [1,2]       # up
    mapping[1,1] = [1,2], [1,1]       # right-up
    mapping[1,-1] = [1,1], [1,0]      # right-down
    mapping[0,-1] = [1,0], [-1,0]     # down
    mapping[-1,-1] = [-1,0], [-1,1]   # left-down
    mapping[-1,1] = [-1,1], [-1,2]    # up-eft


    delete = []
    num_trans = len(transistions) - 1
    if not full:
        for i in range(num_trans):
            current_elem = transistions[i]
            next_elem = transistions[i+1]

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





            # ######
            # up = trans[0,0]%2 == 0
            # sign = 1-2*up
            # diff = trans[1] - trans[0]
            # absdiff = abs(diff)

            # correction = absdiff[0] - (absdiff[0]==absdiff[1])
            # direction = [diff[0], diff[1] + sign * correction]
            # local_pair = mapping[direction[0], direction[1]] # local cordinates to center elem

            # neigh = center_neigh(trans[0])
            # global_pair = neigh[local_pair[:, 0], local_pair[:, 1]] # global atoms coordinates 

            # [delete.append(pair) for pair in global_pair]

    else:
        for trans in transistions:
            global_atoms = center_neigh(trans[0]).astype("int")
            [delete.append(atom) for atom in global_atoms[0]]
            [delete.append(atom) for atom in global_atoms[1]]



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
  
    # Set illegal coordinates to [nan, nan] 
    neigh[np.any(neigh < 0, axis = 2), :] = np.nan 
    return neigh



# def pop_up_pattern():
#     mat = np.ones((20, 40))
#     build_graphene_sheet(mat, view_lattice = True)

#     ref_center = 



if __name__ == "__main__":

    # pop_up_pattern()

    mat = np.ones((5, 12)) # Why does (5, 12) not work?

    transistions = np.array([   [[2,0], [3,1]],
                                [[3,1], [3,2]], 
                                [[3,2], [3,3]],
                                [[3,3], [4,3]],
                                [[4,3], [5,4]] ])


    transistions = np.array([[2,0], [3,1], [3,2], [3,3], [3,4], [4,3], [5,4]])

    delete = center_elem_trans_to_atoms(transistions, full = False)   
    mat[delete[:, 0], delete[:, 1]] = 0

   
    build_graphene_sheet(mat, view_lattice = True)

   




