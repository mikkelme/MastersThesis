from ase.build import graphene_nanoribbon
from ase.build import graphene

from ase.io import  lammpsdata
from ase.visualize import view
import numpy as np




def build_graphene_sheet(mat, view_lattice = False):
    

    Cdis = 1.42 # carbon-carbon distance [Å]

   
    # xlen = mat.shape[0]
    # ylen = mat.shape[1]
    xlen = 3.5
    ylen = 2
  
    # --- Create graphene lattice --- #
    atoms = graphene_nanoribbon(xlen, ylen, type='armchair', saturated=False, C_C=Cdis, vacuum=1.0)
    atoms.pbc = [False, False, False] # Set x,y,z to non periodic (not sure if this is relevant)


    # Swap axes: y <-> z
    new_posistions = atoms.get_positions()[:,(0,2,1)] # swap axis
    new_cell = atoms.get_cell()[(0,2,1),(0,2,1)]

    atoms.set_positions(new_posistions)
    atoms.set_cell(new_cell)


    if view_lattice: 
        view(atoms)



    return

#network X





if __name__ == "__main__":

    mat = np.ones((2,2))
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



