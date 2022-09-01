from build_diamond_block import *
from build_graphene_sheet import *


def build_block_and_sheet(mat, view_lattice = False, write = False):
    object_dis = 15.0 # [Å]

    sheet = build_graphene_sheet(mat, view_lattice = False, write=False)
    minmax_sheet = np.array([np.min(sheet.get_positions(), axis = 0), np.max(sheet.get_positions(), axis = 0)]) # Find min and max positions

    block = build_diamond_block(mat, diamond_thickness = 2, padding = 2, z_shift = object_dis + minmax_sheet[1,2])
    minmax_block = np.array([np.min(block.get_positions(), axis = 0), np.max(block.get_positions(), axis = 0)])  # Find min and max positions

    # --- Translate sheet to align with block center --- #
 

    # Finc centers
    center_sheet = (minmax_sheet[0] + minmax_sheet[1])/2
    center_block = (minmax_block[0] + minmax_block[1])/2

    # Translate sheet atoms
    translation_vec = center_block-center_sheet
    translation_vec[2] = 0
    sheet.translate(translation_vec)

    # Merge sheet and block into same object (uses cell and bc from first object )
    atoms = block + sheet
    # atoms = block # <------------------------ ONLY BLOCK!

    # Write pullblock position to file 
    minmax_sheet += translation_vec
    ylen = minmax_sheet[1,1] - minmax_sheet[0,1]
    pullblock_len = pullblock/mat.shape[1] * ylen

    eps = 1e-3
    yhi = minmax_sheet[0,1] + pullblock_len + eps 
    ylo = minmax_sheet[1,1] - pullblock_len - eps
    zhi = (minmax_sheet[1,2] + minmax_block[0,2])/2
    lim = [yhi, ylo, zhi]
    varname = ['yhi', 'ylo', 'zhi']



    if view_lattice: 
        view(atoms)

    if write:
        lammpsdata.write_lammps_data('./lammps_sheet_and_block', atoms)
        outfile = open('pullblock_lim.in', 'w')
        for i in range(len(lim)):
            outfile.write(f'variable pullblock_{varname[i]} equal {lim[i]}\n') 


    return atoms





    # variable block_elem equal 2 # block thicknes in terms of matrix columns
    # variable yscale equal 1.230 # length per column
    # variable block_thickness equal ${block_elem}*${yscale} # physical length of end blocks
    # variable eps equal 1e-3 # small number for avoiding spill of the freeze region

    # variable ylow_top equal ${block_thickness}-${eps} # max y-val of lower block
    # variable yhigh_bottom equal yhi-${block_thickness}+${eps} # min y val of upper block



    # ==> yhi, ylo, zhi


if __name__ == "__main__":
    # Generate sheet matrix
    multiples = (3, 6)
    multiples = (2, 2)

    unitsize = (5,7)
    mat = pop_up_pattern(multiples, unitsize, view_lattice = False)
    mat, pullblock = build_pull_blocks(mat, pullblock = 6, sideblock = 0)

    # Build block and sheet 
    atoms = build_block_and_sheet(mat, view_lattice = True, write = True)

    
    




  