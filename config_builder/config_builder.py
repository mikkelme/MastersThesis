import sys
sys.path.append('../') # parent folder: MastersThesis
from graphene_sheet.build_graphene_sheet import *

def main(sheet_mat, substrate_file, pullblock = None, view_atoms = False, write = False):
    # Parameters
    sheet_substrate_distance = 6 # [Å]
    bottom_substrate_freeze = 5.5 # [Å]
    eps = 1e-6

    # --- Load atomic structures --- #
    # Add pullblocks
    if pullblock != None:
        sheet_mat, pullblock = build_pull_blocks(sheet_mat, pullblock = pullblock)

    sheet = build_graphene_sheet(sheet_mat, view_lattice = False, write=False)
    substrate = lammpsdata.read_lammps_data(substrate_file, Z_of_type=None, style='atomic', sort_by_id=True, units='metal')
    ##### How to know which ones are frozen!?



    substrate_atomic_num = 14
    substrate.set_atomic_numbers(np.ones(substrate.get_global_number_of_atoms())*substrate_atomic_num) # For visualization
    



    # --- Translate relatively --- #
    # Find min and max positions
    minmax_sheet = np.array([np.min(sheet.get_positions(), axis = 0), np.max(sheet.get_positions(), axis = 0)]) 
    minmax_substrate = np.array([np.min(substrate.get_positions(), axis = 0), np.max(substrate.get_positions(), axis = 0)]) 
    
    # Find centers
    center_sheet = (minmax_sheet[0] + minmax_sheet[1])/2
    center_substrate = (minmax_substrate[0] + minmax_substrate[1])/2

    # Align center and get right distance
    trans_vec1 = center_substrate - center_sheet
    trans_vec1[2] = sheet_substrate_distance - (minmax_sheet[0, 2] - minmax_substrate[1,2])
    sheet.translate(trans_vec1)

    # --- Merge into same object --- #
    merge = sheet + substrate

    # --- Fix cell / simulation box --- #
    # Align with origo
    minmax_merge = np.array([np.min(merge.get_positions(), axis = 0), np.max(merge.get_positions(), axis = 0)]) 
    trans_vec2 = -minmax_merge[0, :] + np.ones(3)*eps
    merge.translate(trans_vec2)
    merge.set_cell(minmax_merge[1,:] + trans_vec2 + np.ones(3)*eps)


    # --- Write information-- #
    # Update sheet and substrate limits
    minmax_sheet += trans_vec1 + trans_vec2 
    minmax_substrate += trans_vec1 + trans_vec2 

    # Pullblock (PB) positions
    PB_len = pullblock/sheet_mat.shape[1] * (minmax_sheet[1,1] - minmax_sheet[0,1])
    yhi = minmax_sheet[0,1] + PB_len + eps 
    ylo = minmax_sheet[1,1] - PB_len - eps
    zhi = (minmax_sheet[1,2] + minmax_substrate[0,2])/2
    PB_lim = [yhi, ylo, zhi]
    PB_varname = ['yhi', 'ylo', 'zhi']

    substrate_freeze_zhi = minmax_substrate[0,2] + bottom_substrate_freeze 



    # Write 
    if view_atoms: 
        view(merge)

    if write:
        lammpsdata.write_lammps_data('./config.txt', merge)
        outfile = open('config_info.in', 'w')

        # Pullblock
        for i in range(len(PB_lim)):
            outfile.write(f'variable pullblock_{PB_varname[i]} equal {PB_lim[i]}\n') 

        # Substrate
        outfile.write(f'variable substrate_freeze_zhi equal {substrate_freeze_zhi}\n') 

    












if __name__ == "__main__":
    multiples = (2, 4)
    unitsize = (5,7)
    mat = pop_up_pattern(multiples, unitsize, view_lattice = False)


    substrate_file = "../substrate/crystal_Si_substrate.txt"
    main(mat, substrate_file, pullblock = 6, view_atoms = False, write = True)