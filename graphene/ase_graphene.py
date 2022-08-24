from ase.build import graphene_nanoribbon
from ase.build import graphene

from ase.io import  lammpsdata
from ase.visualize import view
import numpy as np


def reverse_order(start, end):
    for i in range(start, start + (end-start)//2):
        j = end - i - 1 + start
        print(i, j)
        atoms.positions[[i, j]] = atoms.positions[[j, i]]
    print(" ")
    return



ylen = 2
xlen = 2.5
Cdis = 1.42 # carbon-carbon distance [Å]


# --- Create graphene lattice --- #
atoms = graphene_nanoribbon(xlen, ylen, type='armchair', saturated=False, C_C=Cdis, vacuum=1.0)
atoms.pbc = [False, False, False] # Set x,y,z to non periodic (not sure if this is relevant)



new_positions = atoms.get_positions().copy()
new_cell = atoms.get_cell().copy()


##### Working here #####
print(new_cell)
exit()
#Swap y and z axis
for i, atom in enumerate(atoms):
    new_positions[i] = [atom.position[0], atom.position[2], atom.position[1]]

atoms.set_positions(new_position)
# exit()
# for i, vec in enumerate(atoms.cell):
#     atoms.cell[i] = [vec[0], vec[2], vec[1]]

# print(atoms[0].position)



# --- Change to new coordinate system ---#
# Switch y and z axis
# atoms.rotate(-90, 'x', center=(0,0,0), rotate_cell=True)

# Reorder atoms 
# num_complete_ylines = xlen//1
# y_line_len = 4*ylen #change to ylen

# num_atoms = len(atoms)
# highest_multiple = num_atoms - num_atoms%y_line_len

# reverse_order(0, num_atoms)
# local_reorder = [num_atoms - highest_multiple] + int(num_complete_ylines) * [y_line_len] 

# start = 0
# for loc in local_reorder:
#     print(start, start + loc)
#     reverse_order(start, start + loc)
#     start += loc   

# view(atoms)


lammpsdata.write_lammps_data('./lammps_sheet', atoms)


########################
# for i in range(int(highest_multiple/2)):
#     j = (highest_multiple) - (i//y_line_len + 1)*y_line_len  + i%y_line_len  
#     print(i, j)
#     # atoms.positions[[i, j]] = atoms.positions[[j, i]]

# # Reverse order
# for i in range(num_atoms//2):
#     j = num_atoms - i - 1
#     print(i, j)
#     atoms.positions[[i, j]] = atoms.positions[[j, i]]





# num_atoms = len(atoms)
# for i in range(num_atoms//2):
#     j = num_atoms-i-1
#     print(i, j)
#     atoms.positions[[i, j]] = atoms.positions[[j, i]]


# atoms.translate([0,atoms.cell[2,1] - atoms[0].position[1], 0])
# atoms.rotate(90, 'x', center=(0,0,0), rotate_cell=True)
# 
# print(atoms.cell)

# del_idx = np.arange(20, 38+1)



# for atom in atoms:
#     print(atom)

# for atom in atoms[del_idx]:
#     print(atom)

# del atoms[del_idx]



