import sys
sys.path.append('../') # parent folder: MastersThesis

from graphene_sheet.build_utils import *



def pop_up_pattern(multiples, unitsize = (5,7), sp = 1):
    # --- Parameters --- #
    mat = np.ones((multiples[0]*10, multiples[1]*10)).astype('int') # lattice matrix
    ref = np.array([0, 0]) # reference center element
    size = unitsize # Size of pop_up pattern

    assert (np.abs(size[0] - size[1]) - 2)%4 == 0, f"Unit size = {size} did not fulfill: |size[1]-size[0]| = 2, 4, 6, 10..."
    assert np.min(size) > 0, f"Unit size: {size} must have positives entries."
   
    # --- Set up cut out pattern --- #
    # Define axis for pattern cut out
    m, n = np.shape(mat)
    axis1 = np.array([2*(2 + sp + size[0]//2), 2 + sp + size[0]//2]) # up right
    axis2 = np.array([- 2*(1 + size[1]//2 + sp), 3*(1 + size[1]//2 + sp)]) # up left
    unit2_axis =  np.array([3 + size[0]//2 + size[1]//2, 1 + size[0]//4 + size[1]//4 - size[1]]) + (2*sp, -sp) # 2nd unit relative to ref

 
    # Create unit1 and unit2
    up = ref[0]%2 == 0
    line1 = [ref]
    line2 = []
    
    if up:
        for i in range((size[0]-1)//2):
            line1.append(ref - [i+1, (i+1)//2 ])
            line1.append(ref + [i + 1, i//2 + 1])

        for i in range(sp, size[1] + sp):
            line2.append(ref + [i+1, -(i + (i+1)//2 + 1)])

    else:
        for i in range((size[0]-1)//2):
            line1.append(ref + [i+1, (i+1)//2 ])
            line1.append(ref - [i + 1, i//2 + 1])


        for i in range(sp, size[1] + sp):
            line2.append(ref + [i+1, -(i + i//2 + 2)]) 

    
    del_unit1 = np.array(line1 + line2)
    del_unit2 = np.array(line1 + line2) + unit2_axis


    # --- Translate cut-out-units across lattice --- # 
    # Estimate how far to translate
    range1 = int(np.ceil(np.dot(np.array([m,n]), axis1)/np.dot(axis1, axis1))) + 1      # project top-right corner on axis 1 vector
    range2 = int(np.ceil(np.dot(np.array([0,n]), axis2)/np.dot(axis2, axis2)/2))  + 1   # project top-left corner on axis 2 vector


    # Translate and cut out
    for i in range(range1):
        for j in range(-range2, range2+1):
            vec = i*axis1 + j*axis2 
            del_map1 = del_unit1 + vec
            del_map2 = del_unit2 + vec

            mat = delete_atoms(mat, center_elem_trans_to_atoms(del_map1, full = True))
            mat = delete_atoms(mat, center_elem_trans_to_atoms(del_map2, full = True))

    return mat


def capacitor_line(ref, num_gaps = 3, xlen = 5, xsp = 3, ylen = 9):
    """     
    """
    assert num_gaps % 2 == 1, "num_gaps must be odd"
    assert xlen % 2 == 1, "xlen must be odd"
    assert ylen % 2 == 1, "ylen must be odd"
    assert xsp % 2 == 1, "xsp must be odd"
    
    assert xlen > 0, "xlen must be > 0" 
    assert ylen > 0, "ylen must be > 0" # Maybe >= 3 for it to make sense
    assert ysp > 0, "ysp must be > 0" 
    
    
    delmap = []
    for k in range(-num_gaps//2, 1+num_gaps//2):
        xmod = xsp // 2 + 1
        
        start_pos = ref + [2*k*(xlen+2*xmod-1), 0]
        working_pos = start_pos + [2*(xsp//2+1), 0]
        
        # # Draw horisontal lines connection capacitor shapes
        for i in range(xlen):
            delmap.append(working_pos + [2*i, 0])
        
        
        # Draw vertical capacitor shapes
        for j in range(1, 1+(ylen-1)//2):
            delmap.append(working_pos + [0, j])
            delmap.append(working_pos - [0, j])
            delmap.append(working_pos + [2*(xlen-1), j])
            delmap.append(working_pos - [-2*(xlen-1), j])
            
            
            
        
        
        
    
    return delmap

def honeycomb():
    """ Inspired by Scotch Cushion Lock Protective Wrap """
    
    
    # mat = np.ones((50, 90)).astype('int') # lattice matrix
    
    # mat = np.ones((100, 200)).astype('int') # lattice matrix
    # ref = np.array([50, 50]) # reference center element
    
    mat = np.ones((50, 100)).astype('int') # lattice matrix
    ref = np.array([25, 25]) # reference center element
    
    # Honeycomb variables
    xwidth = 3 
    ywidth = 3  
    bridge_thickness = 1
    bridge_len = 9 
    ####################
    
    assert xwidth%2 == 1, "xwidth must be odd"
    assert bridge_thickness%2 == 1, "bridge_thickness must be odd"
    assert bridge_len%2 == 1, "bridge_len must be odd"
    
    
    
    
    # Capacitor lines parameters 
    xlen = xwidth + bridge_thickness + 4 + 1
    ylen = bridge_len
    xsp = bridge_thickness
    
    num_gaps = 9
    num_lines = 9
    
    
    trans_ver = (1 + ylen//2 + ywidth + 1)
    for even in range(0, num_lines, 2):
        ref_plus = ref + [0, even*trans_ver]
        ref_neg = ref - [0, even*trans_ver]
        
        delmap = capacitor_line(ref_plus, num_gaps, xlen, xsp, ylen)
        delmap += capacitor_line(ref_neg, num_gaps, xlen, xsp, ylen)
        delete_atoms(mat, center_elem_trans_to_atoms(delmap, full = True))
        
    
    for odd in range(1, num_lines, 2):
        ref_plus = ref + [(2*(1 + xsp//2 + xlen//2)), odd*trans_ver]
        ref_neg = ref - [(2*(1 + xsp//2 + xlen//2)), odd*trans_ver]
        
        delmap = capacitor_line(ref_plus, num_gaps, xlen, xsp, ylen)
        delmap += capacitor_line(ref_neg, num_gaps, xlen, xsp, ylen)
        delete_atoms(mat, center_elem_trans_to_atoms(delmap, full = True))

    
    
    
    # Reverse for visulaization purposes
    # mat[mat == 1] = 2
    # mat[mat == 0] = 1
    # mat[mat == 2] = 0
    return mat    


def stitch_cuts(commands):
    obj = []
    for i, com in enumerate(commands):
        print(i, com)
    exit()

def half_octans():
    mat = np.ones((20, 40))
    full = False
    sp = 2 # > 0

    ver = 4
    diag = 5 # = odd
    hor = 5 # odd



    ref = np.array([2,9])
    # if up:

    # + ref
    zero = np.array([0,0])
    up = [zero + [0, i] for i in range(ver)]
    up_right = [zero + [i, (i+1)//2] for i in range(diag)]
    right = [zero + [i, 0] for i in range(hor)]
    down_right = [zero + [i, -(i//2) ] for i in range(diag)]
    down = [zero - [0, i] for i in range(ver)]

    commands = [up, up_right, right, down_right, down]
    stitch_cuts(commands)

    top = np.concatenate((up, up_right + up[-1], )) + ref
    # top = np.array(up + up_right + right + down_right + down) + ref

    # down = np.array(ver_list) + ref + [sp*2, 0]

    # down[:,0] += sp*2

    print(top)
    print()
    # print(down)
    # exit()
    
    # print(ver_list)
    # print(diag_list)
    # print(hor_list)
    # print(downdiag_list)
    # print(downver_list)
    # exit()

    # mat = delete_atoms(mat, center_elem_trans_to_atoms(ver1, full = full))    
    mat = delete_atoms(mat, center_elem_trans_to_atoms(top, full = full))    
    mat = delete_atoms(mat, center_elem_trans_to_atoms(down, full = full))    
    # mat = delete_atoms(mat, center_elem_trans_to_atoms(ver2, full = full))    
    # mat = delete_atoms(mat, center_elem_trans_to_atoms(ver3, full = full))    
    # mat = delete_atoms(mat, center_elem_trans_to_atoms(ver4, full = full))    


    return mat




if __name__ == "__main__":
    pass
    # mat = pop_up_pattern((3,3))
