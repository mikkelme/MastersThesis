import sys
sys.path.append('../') # parent folder: MastersThesis

from graphene_sheet.build_utils import *
import random

def pop_up(shape = (62, 106), size = (5,3), sp = 1, ref = None):
    """ Generate pop-up cut patteren inspired by:
        https://seas.harvard.edu/news/2017/02/new-pop-strategy-inspired-cuts-not-folds

    Args:
        shape (tuple, optional): Shape of cut matrix. Defaults to (50, 100).
        size (tuple, optional): Shape of repeated cut pattern. Defaults to (5,3).
        sp (int, optional): Spacing between cuts. Defaults to 1.
        ref (array, optional): Starting point for the cut pattern. Defaults to center of matrix.

    Returns:
        2D array: Cut pattern matrix
    """  
    
    # Build cut matrix
    mat = np.ones((shape[0], shape[1])).astype('int')
    
    # Reference position
    if ref is None: # Defaults to center
        ref = np.array([mat.shape[0]//2, mat.shape[1]//4]) 
    elif isinstance(ref, str):
        if ref == 'RAND':
            ref = np.array((random.randint(0, mat.shape[0]), random.randint(0, mat.shape[1]//2)))
        else:
            print(f'REF = {ref} is not understood.')
            exit()
    else:
        ref = np.array(ref)
  
    # Catch unvalid settings
    assert size[0]%2 == 1 and size[1]%2 == 1, f"Unit size = {size} must only contain odd numbers"
    assert (np.abs(size[0] - size[1]) - 2)%4 == 0, f"Unit size = {size} did not fulfill: |size[1]-size[0]| = 2, 6, 10..."
    assert np.min(size) > 0, f"Unit size: {size} must have positives entries."
   
    # --- Set up cut out pattern --- #
    # Define axis for pattern cut out
    m, n = np.shape(mat)
    axis1 = np.array([2*(2 + sp + size[0]//2), 2 + sp + size[0]//2])        # up right
    axis2 = np.array([- 2*(1 + size[1]//2 + sp), 3*(1 + size[1]//2 + sp)])  # up left
    unit2_axis =  np.array([3 + size[0]//2 + size[1]//2, 1 + size[0]//4 + size[1]//4 - size[1]]) + (2*sp, -sp) # 2nd unit relative to ref


    # Create unit1 and unit2
    up = ref[0]%2 == 0
    line1 = [ref]
    line2 = []
    
    if up: # If reference is on a 'up' column
        for i in range((size[0]-1)//2):
            line1.append(ref - [i+1, (i+1)//2 ])
            line1.append(ref + [i + 1, i//2 + 1])

        for i in range(sp, size[1] + sp):
            line2.append(ref + [i+1, -(i + (i+1)//2 + 1)])

    else: # If reference is on a 'down' column
        for i in range((size[0]-1)//2):
            line1.append(ref + [i+1, (i+1)//2 ])
            line1.append(ref - [i + 1, i//2 + 1])

        for i in range(sp, size[1] + sp):
            line2.append(ref + [i+1, -(i + i//2 + 2)]) 

    
    del_unit1 = np.array(line1 + line2)
    del_unit2 = np.array(line1 + line2) + unit2_axis


    # --- Translate cut-out-units across lattice --- # 
    # Estimate how far to translate to cover the whole sheet
    range1 = int(np.ceil(np.dot(np.array([m,n]), axis1)/np.dot(axis1, axis1))) + 1      # project top-right corner on axis 1 vector
    range2 = int(np.ceil(np.dot(np.array([0,n]), axis2)/np.dot(axis2, axis2)/2))  + 1   # project top-left corner on axis 2 vector

    # Number of possible unique pertubations of ref
    M = int(np.abs(np.cross(axis1, axis2))/2)
    
    # Translate and cut out
    for i in range(-range1, range1+1):
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
    assert xsp > 0, "xsp must be > 0" 
    
    
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

def honeycomb(shape = (62, 106), xwidth = 1, ywidth = 1,  bridge_thickness = 1, bridge_len = 5, ref = None):
    """ Generate honeycomb cut pattern inspired by:
        Scotch Cushion Lock Protective Wrap
    

    Args:
        shape (tuple, optional): Shape of cut matrix. Defaults to (50, 100).
        xwidth (int, optional): Width of lines going in y-direction. Defaults to 1.
        ywidth (int, optional): Width of lines going in x-direction . Defaults to 1.
        bridge_thickness (int, optional): Thickness of bridge connection rows. Defaults to 1.
        bridge_len (int, optional): Length of bridge connection rows. Defaults to 5.
        ref (array, optional): Starting point for the cut pattern. Defaults to center of matrix.

    Returns:
        2D array: Cut pattern matrix
    """
    
    # Build cut matrix
    mat = np.ones((shape[0], shape[1])).astype('int') 
    

    # Reference position
    if ref is None: # Defaults to center
        ref = np.array([mat.shape[0]//2, mat.shape[1]//4]) 
    elif isinstance(ref, str):
        if ref == 'RAND':
            ref = np.array((random.randint(0, mat.shape[0]), random.randint(0, mat.shape[1]//2)))
        else:
            print(f'REF = {ref} is not understood.')
            exit()
    else:
        ref = np.array(ref)
  
  
   
    # Catch unvalid settings
    assert xwidth%2 == 1, "xwidth must be odd"
    assert bridge_thickness%2 == 1, "bridge_thickness must be odd"
    assert bridge_len%2 == 1, "bridge_len must be odd"
    
    
    # Capacitor lines parameters 
    xlen = xwidth + bridge_thickness + 4 + 1
    ylen = bridge_len
    xsp = bridge_thickness
    
    # Vertical translation
    trans_ver = (1 + ylen//2 + ywidth + 1)
    
    # Estimate how far to translate
    num_lines = int(np.ceil((mat.shape[1]//2) / trans_ver))
    num_gaps = int((np.ceil(2*(mat.shape[0] / (2*(xlen+ 2*(xsp//2 + 1) -1))))) // 2 * 2 + 1)
    
    # --- Build and translate capacitor (looking) lines --- #
    # Even rows
    for even in range(0, num_lines, 2):
        ref_plus = ref + [0, even*trans_ver]
        ref_neg = ref - [0, even*trans_ver]
        
        delmap = capacitor_line(ref_plus, num_gaps, xlen, xsp, ylen)
        delmap += capacitor_line(ref_neg, num_gaps, xlen, xsp, ylen)
        delete_atoms(mat, center_elem_trans_to_atoms(delmap, full = True))
        
    # Odd rows
    for odd in range(1, num_lines, 2):
        ref_plus = ref + [(2*(1 + xsp//2 + xlen//2)), odd*trans_ver]
        ref_neg = ref - [(2*(1 + xsp//2 + xlen//2)), odd*trans_ver]
        
        delmap = capacitor_line(ref_plus, num_gaps, xlen, xsp, ylen)
        delmap += capacitor_line(ref_neg, num_gaps, xlen, xsp, ylen)
        delete_atoms(mat, center_elem_trans_to_atoms(delmap, full = True))

    return mat    




# def unique_mat(shape, list):
#     matrices = [] 
#     for ref in list:
#         mat = pop_up(shape, np.array(ref))
#         if len(matrices) == 0:
#             matrices.append(mat)      
#             print(f'unique ref = {ref}')
#         elif not np.any(np.all(np.all(mat == matrices, axis = 1), axis = 1)):
#             matrices.append(mat)
#             print(f'unique ref = {ref}')
#         else:
#             print(f'dubplicate ref = {ref}')
            
    
#     print(f'unique = {len(matrices)}/{len(list)}')
    
    
def all_unique_mat(shape):
    matrices = [] 
 
    A = shape[0]
    B = shape[1]//2
    
    count = 0
    for i in range(A):
        for j in range(B):
            mat = pop_up(shape, ref = np.array((i,j)))
            
            if len(matrices) == 0:
                matrices.append(mat)
                
            elif not np.any(np.all(np.all(mat == matrices, axis = 1), axis = 1)):
                matrices.append(mat)
            
            count += 1
            print(f'ref = ({i}, {j}), unique = {len(matrices)}/{count}, tested = {i*B + j + 1}/{A*B} ')
            
            
    

if __name__ == "__main__":
    
    
    # unique_mat([(24, 25), (25, 26), (26,26), (27,27), (28,27), (29,28), (30,28), (25,25)])
    # unique_mat([(24, 25), (25,25), (25, 26), (26,26), (27,27), (28,27), (29,28), (30,28)])
    # unique_mat([16, 32], [(0, 0), (0,1), (0, 2), (0, 6)])
    
    # matrices = []
    # mat0 = pop_up([16, 32], np.array((0, 0)))
    # mat1 = pop_up([16, 32], np.array((0, 1)))
    # mat2 = pop_up([16, 32], np.array((0, 2)))
    
    # matrices.append(mat0)
    # matrices.append(mat1)
    # # matrices.append(mat2)
    
    # test = not np.any(np.all(np.all(mat2 == matrices, axis = 1), axis = 1))
    # print(test)
    # print(np.any(np.all(np.all(mat1 == matrices, axis = 1), axis = 1)))
    
    # test = np.all(mat0 == matrices, axis = 1)
    # test1 = np.all(test, axis = 1)
    # print(np.any(test1))
    # print()
    
    
    
    
    ref = np.array((0,0))
    a = np.array((6, 3))
    b = np.array((-6, 9))
    c = np.array((6, -3))
     
    M = int(np.abs(np.cross(a, b))/2)
    print(M)
    # # all_unique_mat((16, 32))
    
    
   
                
