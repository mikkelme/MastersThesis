from ase.build import graphene_nanoribbon
from ase.build import graphene

from ase.io import lammpsdata
from ase.spacegroup import crystal
from ase.visualize import view
import numpy as np



def build_diamond_block(mat, diamond_thickness = 2, padding = 2, z_shift = 3*3.57):
    # Consider adding eps to avod atoms on cell edge!
    m, n = np.shape(mat)

    
    vacuum = 2.0
    a = 2.460
    a1 = np.array([0, a, 0])
    a2 = np.array([a*np.sqrt(3)/2, -a/2, 0])
    b = 2/3*a1 + 1/3*a2 + vacuum

    xmax = b[0] + a*np.sqrt(3)/2 * (m-1) 
    ymax = 1/2*a * (n-1)

    diamond_spacing = 3.57
    xlen = int((xmax//diamond_spacing + 1)) + padding
    ylen = int((ymax//diamond_spacing + 1)) + padding 


    # xlen, ylen, zlen = (1,2,1) # <---------------------- REMOVE
    diamond = crystal('C', [(0,0,0)], spacegroup=227, cellpar=[3.57, 3.57, 3.57, 90, 90, 90], size=(xlen, ylen, diamond_thickness), pbc = False)
    diamond.translate([0,0,z_shift])

    new_cell = diamond.cell
    new_cell[2,2] += z_shift
    diamond.set_cell(new_cell)
    
    return diamond








if __name__ == "__main__":
    mat = np.ones((10,10)) # for sheet
    diamond = build_diamond_block(mat)
    view(diamond)
    # lammpsdata.write_lammps_data('./lammps_diamond_block', diamond)


