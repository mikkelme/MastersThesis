from indexing import *

def plot_sheet(mat, ax, radius, **param):
    full = build_graphene_sheet(np.ones(np.shape(mat)))
    
    pos = full.get_positions().reshape(*np.shape(mat), 3)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0.5:
                x, y = pos[i, j, 0:2]
                circle = plt.Circle((x, y), radius,  **param)
                ax.add_patch(circle)
                
    ax.axis('equal')
    
    
def plot_center_coordinates(shape, ax, radius, **param):
    # Build dummy sheet
    full = build_graphene_sheet(np.ones(shape))
    pos = full.get_positions().reshape(*shape, 3)
    
    # Get corners
    BL = pos[0, 0, 0:2] # Bottom left
    TR = pos[-1, -1, 0:2] # Top right

    # Translation axis
    Cdis = 1.461
    a = 3*Cdis/np.sqrt(3)
    Bx = a/(np.sqrt(3)*2)
    vecax = a*np.sqrt(3)/2
    vecay = a/2
    Lx = (vecax + Bx)/2 
    
    xs, ys = BL[0], BL[1] # Start
    for j in range(0, shape[1], 2):
        for i in range(0, shape[0] + 1):
            x = xs + (1+ 3/2*(i - 1)) * Lx
            y = ys + a*(1/2 + 1/2*j - 1/2*(i%2)) 
            circle = plt.Circle((x, y), radius, **param)
            ax.add_patch(circle)
            
    
  

def show_pop_up(save = False):
    # Settings
    shape = (40,50)
    ref = np.array([shape[0]//2+1, shape[1]//4])
    sp = 2
    size = (7,5)
    atom_radii = 0.6
    center_radii = 0.2

   
    # --- Get configuration --- #
    mat = pop_up(shape, size, sp, ref = ref)
    
    
    # --- Get cut units --- #
    unit1 = np.ones(np.shape(mat))
    unit2 = np.ones(np.shape(mat))
    
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
    
    del_hor = np.concatenate((np.array(line1), np.array(line1) + unit2_axis))
    del_ver = np.concatenate((np.array(line2), np.array(line2) + unit2_axis))
    unit_hor = np.ones(shape)
    unit_ver = np.ones(shape)
    
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
            unit_hor =  delete_atoms(unit_hor, center_elem_trans_to_atoms(del_hor + vec, full = True))
            unit_ver =  delete_atoms(unit_ver, center_elem_trans_to_atoms(del_ver + vec, full = True))
    
    
    unit1 = delete_atoms(np.ones(shape), center_elem_trans_to_atoms(del_unit1, full = True))
    unit2 = delete_atoms(np.ones(shape), center_elem_trans_to_atoms(del_unit2, full = True))
    
    unit1 = 1 - unit1
    unit2 = 1 - unit2
    unit_hor = 1 - unit_hor
    unit_ver = 1 - unit_ver
    unit1_hor = np.where(np.logical_and(unit1 == unit_hor, unit1 == 1), 1, 0)
    unit1_ver = np.where(np.logical_and(unit1 == unit_ver, unit1 == 1), 1, 0)
    
    # --- Get space indicators --- #
    up = ref[0]%2 == 0
    sp1 = 1-delete_atoms(np.ones(shape), center_elem_trans_to_atoms([ref + [size[0]//2+(i+1), (size[0]//2 + (i+2-ref[0]%2))//2] for i in range(sp+1)], full = True))
    sp2 = 1-delete_atoms(np.ones(shape), center_elem_trans_to_atoms([ref + [i+1, -(i+1) - (i+1+ref[0]%2)//2] for i in range(sp)], full = True))
    
    # --- Plot --- # 
    green  = color_cycle(3)
    orange = color_cycle(4)
    blue = color_cycle(6)
    
    # Remove overlap
    sp1[1-mat == 1] = 0
    
    
    # --- Inverse --- #
    fig1 = plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    fig2 = plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k')
    ax1 = fig1.gca()
    ax1.set_facecolor("white")
    ax2 = fig2.gca()
    ax2.set_facecolor("white")

    
    
    # fig, axes = plt.subplots(1, 2, num = unique_fignum(), figsize = (10,5))
    plot_sheet(unit1_hor, ax1, atom_radii, facecolor = shade_color(green, 1.3), edgecolor = 'black')
    plot_sheet(unit1_ver, ax1, atom_radii, facecolor = shade_color(orange, 1.3), edgecolor = 'black')
    plot_sheet(unit_hor - unit1, ax1, atom_radii, facecolor = shade_color(green, 1.5), edgecolor = 'black', alpha = 0.6)
    plot_sheet(unit_ver - unit1, ax1, atom_radii, facecolor = shade_color(orange, 1.5), edgecolor = 'black', alpha = 0.6)
    
    # Background
    plot_sheet(mat, ax1, atom_radii, facecolor = 'None', edgecolor = 'black', alpha = 0.2)
    
    # Spacing 
    plot_sheet(sp1, ax1, atom_radii, facecolor = blue, edgecolor = 'black', alpha = 0.2)
    plot_sheet(sp2, ax1, atom_radii, facecolor = blue, edgecolor = 'black', alpha = 0.2)
    
    
    # --- Pattern --- #
    plot_sheet(mat, ax2, atom_radii, facecolor = 'grey', edgecolor = 'black')
    plot_sheet(1-mat, ax2, atom_radii, facecolor = 'None', edgecolor = 'black', alpha = 0.2)
    
    # Center coordinates for both
    plot_center_coordinates(np.shape(mat), ax1, center_radii, facecolor = shade_color(blue, 1.5), edgecolor = None)
    plot_center_coordinates(np.shape(mat), ax2, center_radii, facecolor = blue, edgecolor = None)
    
    
    # Remove grid and ticks
    ax1.grid(False)
    ax2.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Set axies
    fig1.supxlabel(r"$x$ (armchair direction)", fontsize = 14)
    fig1.supylabel(r"$y$ (zigzag direction)", fontsize = 14)
    fig1.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    fig2.supxlabel(r"$x$ (armchair direction)", fontsize = 14)
    fig2.supylabel(r"$y$ (zigzag direction)", fontsize = 14)
    fig2.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    
    if save:
        fig1.savefig('../article/figures/system/pop_up_inverse.pdf', bbox_inches='tight')
        fig2.savefig('../article/figures/system/pop_up_pattern.pdf', bbox_inches='tight')

    
def show_honeycomb(save = False):
    pass   
    
   

if __name__ == '__main__':
    show_pop_up(save = True)
    plt.show()