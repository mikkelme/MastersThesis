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


def center_elem_trans_to_atom(transistions):
    """ Remove atom paor when crossing to a new center element """

    # [down, left-down, left-up, up, right-up, right-down]

    # mapping = np.zeros((3,3))
    # mapping[0,1] = 1
    # mapping[1,1] = 2
    # mapping[1,-1] = 3
    # mapping[0,-1] = 4
    # mapping[-1,-1] = 5
    # mapping[-1,1] = 6

    mapping = np.zeros((3,3, 2, 2))
    mapping[0,1] = [[0,2], [1,2]]
    mapping[1,1] = [[1,2], [1,1]]
    mapping[1,-1] = [[1,1], [1,0]]
    mapping[0,-1] = [[1,0], [-1,0]]
    mapping[-1,-1] = [[-1,0], [-1,1]]
    mapping[-1,1] = [[-1,1], [-1,2]]



    




    for trans in transistions:
        up = trans[0,0]%2 == 0
        sign = 1-2*up
        diff = trans[1] - trans[0]
        absdiff = abs(diff)

        correction = absdiff[0] - (absdiff[0]==absdiff[1])
        direction = [diff[0], diff[1] + sign * correction]


        print(mapping[direction[0], direction[1]])

     




def center_neigh(center_elem):
    neigh = np.zeros((2,3,2))
    n, m = center_elem

    # m_start = max(2*m -n%2,0)
    # m_end = 2*m -n%2 + 2 

    m_start = 2*m -n%2
    m_end = m_start + 2 
    for i in range(2):
        for j in range(3):
            neigh[i, j] = [n-1 + i, m_start + j]
  
    # Set illegal coordinates to [nan, nan] 
    neigh[np.any(neigh < 0, axis = 2), :] = np.nan 
    return neigh





if __name__ == "__main__":
    mat = np.ones((5, 10)) # Why does (5, 12) not work?
    # mat[(0, 0, 1, 1, 2, 2, 3, 3, 4, 4), (3, 4, 4, 5, 5, 6, 6, 7, 7, 8)] = 0
    # build_graphene_sheet(mat, view_lattice = True)

    # center_elem = [0,0]
    # coords = center_neigh(center_elem)


    # transistions = np.array([[[2,0], [2,1]], [[2,0], [3,1]], [[2,0], [3,0]], [[2,0], [2,-1]], [[2,0], [1,0]], [[2,0], [1,1]]])
    # transistions = np.array([[[3,1], [3,2]], [[3,1], [4,1]], [[3,1], [4,0]], [[3,1], [3,0] ], [[3,1], [2,0]], [[3,1], [2,1]], [[3,1], [3,2]] ])

    transistions = np.array([[[2,0], [3,1]]])

    center_elem_trans_to_atom(transistions)    







# del_idx = np.arange(20, 38+1)



# for atom in atoms:
#     print(atom)

# for atom in atoms[del_idx]:
#     print(atom)

# del atoms[del_idx]



