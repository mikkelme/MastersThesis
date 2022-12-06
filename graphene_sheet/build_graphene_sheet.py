import sys
sys.path.append('../') # parent folder: MastersThesis

import ase # go through imports to clean up

from ase.build import graphene_nanoribbon
# from ase.build import graphene

from ase.io import  lammpsdata
from ase.visualize import view
from ase.io import write
import numpy as np

from graphene_sheet.manual_patterns import *
from graphene_sheet.RN_walks import *

import os

def build_graphene_sheet(mat, view_lattice = False, write_file = False):
    Cdis = 1.42 # carbon-carbon distance [Å]
    
    shape_error = f"SHAPE ERROR: Got matrix of shape {np.shape(mat)}, y-axis must be multiple of 2 and both nonzero integer."
    assert mat.shape[0]%1 == 0 and mat.shape[1]%1 == 0 and mat.shape[1]%2 == 0, shape_error
    assert mat.shape[0] > 0 and mat.shape[1] > 0, shape_error

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

    #--- Reorder atoms ---# (Is this important)
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

    if write_file != False:
        lammpsdata.write_lammps_data(write, atoms)

    # write('../Presentation/Images/cutpattern.png', atoms)
    return atoms


def save_mat(mat, folder):
    file_id = 1
    
    # Existing data without extension
    existing_data = [o.split(".")[0] for o in os.listdir(folder)] 

    # Generate unique filename    
    filename = f"tmp{file_id}"
    while filename in existing_data:
        file_id += 1
        filename = f"tmp{file_id}"
    
    # Save matrix as array
    np.save(os.path.join(folder, filename), mat)
   

   

if __name__ == "__main__":
    # multiples = (4, 5)
    multiples = (5, 10)
    unitsize = (5,7)
    mat = pop_up_pattern(multiples, unitsize, sp = 2)

    # mat[:] = 1
    # save_mat(mat, "test_data")
    
    # exit()
    
    # RN = RN_Generator( size = (50,50), 
    #                    num_walks = 9,
    #                    max_steps = 15,
    #                    max_dis = 1,
    #                    bias = [(1,1), 0.5],
    #                    periodic = True,
    #                    avoid_unvalid = False,
    #                    grid_start = True,
    #                    center_elem = True)
    
    # mat = RN.generate()
    
    
    # mat = RN.valid
    # mat[mat == 1] = 2
    # mat[mat == 0] = 1
    # mat[mat == 2] = 0
    
    
    # mat, pullblock = build_pull_blocks(mat, pullblock = 6, sideblock = 0)
    build_graphene_sheet(mat, view_lattice = True)






   


   




