from build_config import *


def build_sheet_vaccum(sheet_mat, pullblock = None):
    # Parameters
    eps = 1e-6

    if pullblock != None:
        sheet_mat, pullblock = build_pull_blocks(sheet_mat, pullblock = pullblock)

    build_graphene_sheet(sheet_mat, view_lattice = True, write = './sheet_vacuum.txt')
    sheet = build_graphene_sheet(sheet_mat, view_lattice = False, write = './sheet_vacuum.txt')

    # --- Write information-- #
    # min/max coordinates
    minmax_sheet = np.array([np.min(sheet.get_positions(), axis = 0), np.max(sheet.get_positions(), axis = 0)]) 

    # Pullblock (PB) positions
    PB_len = pullblock/sheet_mat.shape[1] * (minmax_sheet[1,1] - minmax_sheet[0,1])
    PB_yhi = minmax_sheet[0,1] + PB_len + eps 
    PB_ylo = minmax_sheet[1,1] - PB_len - eps
    PB_lim = [PB_yhi, PB_ylo]
    PB_varname = ['yhi', 'ylo']

 
    # Pullblock
    outfile = open('info_sheet_vacuum.in', 'w')
    for i in range(len(PB_lim)):
        outfile.write(f'variable pullblock_{PB_varname[i]} equal {PB_lim[i]}\n') 




if __name__ == "__main__":
    multiples = (4, 5)
    unitsize = (5,7)
    mat = pop_up_pattern(multiples, unitsize, sp = 2, view_lattice = False)
    build_sheet_vaccum(mat, pullblock = 6)
