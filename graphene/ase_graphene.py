from ase.build import graphene_nanoribbon
from ase.build import graphene

from ase.io import  lammpsdata
from ase.visualize import view
import numpy as np


def reverse_order(atoms, start, end):
    for i in range(start, start + (end-start)//2):
        j = end - i - 1 + start
        atoms.positions[[i, j]] = atoms.positions[[j, i]]
    return



def build_graphene_sheet(mat, view_lattice = False):
    Cdis = 1.42 # carbon-carbon distance [Å]


    shape_error = f"SHAPE ERROR: Got matrix of shape {np.shape(mat)}, y-axis-len must be mut multiple of 4 and both nonzero."
    assert mat.shape[0]%1 == 0 and mat.shape[1]%1 == 0 and mat.shape[1]%4 == 0, shape_error
    assert mat.shape[0] != 0 and mat.shape[1] != 0, shape_error

   
    xlen = mat.shape[0]
    ylen = mat.shape[1]//4

  
    # --- Create graphene lattice --- #
    atoms = graphene_nanoribbon(xlen, ylen, type='armchair', saturated=False, C_C=Cdis, vacuum=1.0)
    atoms.pbc = [False, False, False] # Set x,y,z to non periodic (not sure if this is relevant)


    # Swap axes: y <-> z
    new_posistions = atoms.get_positions()[:,(0,2,1)] # swap axis
    new_cell = atoms.get_cell()[(0,2,1),(0,2,1)]

    atoms.set_positions(new_posistions)
    atoms.set_cell(new_cell)


    #--- Reorder atoms ---# (Increasing in y-dir and then x-dir) 
    num_complete_ylines = xlen//1
    yline_len = 4*ylen 

    num_atoms = len(atoms)
    highest_multiple = num_atoms - num_atoms%yline_len

    reverse_order(atoms, 0, num_atoms)
    local_reorder = [num_atoms - highest_multiple] + int(num_complete_ylines) * [yline_len] 

    start = 0
    for loc in local_reorder:
        reverse_order(atoms, start, start + loc)
        start += loc   


    for i in reversed(range(mat.shape[0])):
        for j in reversed(range(mat.shape[1])):
            if mat[i,j] == 0:
                print(atoms[i*yline_len+j].index)
                del atoms[i*yline_len+j]

    if view_lattice: 
        view(atoms)

    # if write
    lammpsdata.write_lammps_data('./lammps_sheet', atoms)


    return

#network X





if __name__ == "__main__":
    mat = np.random.randint(0,2,(4,4))
    mat = np.ones((2,20))
    print(mat)
    build_graphene_sheet(mat, view_lattice = True)
#    mat[0,:4] = 0
#    mat[1,4:] = 0
#    mat[2,:4] = 0








# del_idx = np.arange(20, 38+1)



# for atom in atoms:
#     print(atom)

# for atom in atoms[del_idx]:
#     print(atom)

# del atoms[del_idx]



