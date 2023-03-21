from indexing import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_sheet(mat, ax, radius, **param):
    full = build_graphene_sheet(np.ones(np.shape(mat)))
    
    pos = full.get_positions().reshape(*np.shape(mat), 3)
    on = mat > 0.5
    xmin = np.min(pos[on, 0])
    ymin = np.min(pos[on, 1])
    xmax = np.max(pos[on, 0])
    ymax = np.max(pos[on, 1])
    

    
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if on[i, j]:
                x, y = pos[i, j, 0:2]
                circle = plt.Circle((x, y), radius,  **param)
                ax.add_patch(circle)
                
    ax.axis('equal')
    
    return xmin, ymin, xmax, ymax
    
    
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
            ax.add_patch(circle) # direct way

            
    return BL, TR

def show_pop_up(save = False):
    # Settings
    shape = (40,50)
    ref = np.array([shape[0]//2, shape[1]//4])
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
    fig1 = plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k'); ax1 = fig1.gca()
    fig2 = plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k'); ax2 = fig2.gca()
    ax1.set_facecolor("white")
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
    plot_center_coordinates(np.shape(mat), ax1, center_radii, facecolor = blue, edgecolor = None)
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
        
        
def pop_up_flavors(save = False):
    # Settings
    shape = (40,80)
    patterns = [ (9, 3, 4), (3, 9, 3), (3, 5, 1), (3, 1, 1)]
    
    atom_radii = 0.6
    center_radii = 0.2
    blue = color_cycle(6)
    
    fig, axes = plt.subplots(1, 4, num = unique_fignum(), figsize = (12,4))
    # fig, axes = plt.subplots(1, 4, num = unique_fignum(),  dpi=80, facecolor='w', edgecolor='k')
    
    for i, p in enumerate(patterns):
        mat = pop_up(shape, (p[0], p[1]), p[2], ref = None)
        ax = axes[i]
        name = f'{p}'
        # ax = axes[i//axes.shape[1], i%axes.shape[1]]
        
        plot_sheet(mat, ax, atom_radii, facecolor = 'grey', edgecolor = 'black') # Pattern   
        plot_sheet(1-mat, ax, atom_radii, facecolor = 'None', edgecolor = 'black', alpha = 0.2)  # Background
        plot_center_coordinates(np.shape(mat), ax, center_radii, facecolor = blue, edgecolor = None) # Center elements
        
        # plot settings
        ax.set_title(name)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        
    
    # Set axies
    fig.supxlabel(r"$x$ (armchair direction)", fontsize = 14)
    fig.supylabel(r"$y$ (zigzag direction)", fontsize = 14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
   
    if save:
        fig.savefig('../article/figures/system/pop_up_flavors.pdf', bbox_inches='tight')
    
def show_honeycomb(save = False):
    # Settings
    shape = (40,50)
    xwidth = 3
    ywidth = 2
    bridge_thickness = 1
    bridge_len = 5
    ref = np.array([shape[0]//2, shape[1]//4]) 
    atom_radii = 0.6
    center_radii = 0.2
    
    mat = honeycomb(shape, xwidth, ywidth, bridge_thickness, bridge_len, ref)
    
    fig1 = plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k'); ax1 = fig1.gca()
    fig2 = plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k'); ax2 = fig2.gca()
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")
    
    
    
    # --- Objects --- #
    bridge = 1-delete_atoms(np.ones(shape), center_elem_trans_to_atoms([[ref[0] + 2*(i-bridge_thickness//2), ref[1] + j - bridge_len//2]  for i in range(bridge_thickness) for j in range(bridge_len)], full = True))
    yw = 1-delete_atoms(np.ones(shape), center_elem_trans_to_atoms([[ref[0]+i, j+ref[1]+bridge_len//2+1 + (1-ref[0]%2)] for i in range(-(3+2*(bridge_thickness//2)),(3+2*(bridge_thickness//2))+1, 2) for j in range(ywidth)], full = True))
    # xw = 1-delete_atoms(np.ones(shape), center_elem_trans_to_atoms([[ref[0]+i, j+ref[1]+bridge_len//2+1 + (1-ref[0]%2)] for i in range((3+2*(bridge_thickness//2)),(3+2*(bridge_thickness//2))+2+xwidth , 2) for j in range(ywidth)], full = True))
    xw = 1-delete_atoms(np.ones(shape), center_elem_trans_to_atoms([ [ref[0]+i + 4 + 2*(bridge_thickness//2), ref[1]-1-j+(ref[0]+i)%2] for i in range(xwidth) for j in range(ywidth)], full = True))
    
    # Remove left and right column from yw
    yw[np.min(np.argwhere(yw == 1)[:, 0]), :] = 0;
    yw[np.max(np.argwhere(yw == 1)[:, 0]), :] = 0;
    
    
    # Remove overlap
    yw[1-mat == 1] = 0
    xw[1-mat == 1] = 0
    xw[yw == 1] = 0

    
    # Remaning cuts
    remaning = 1-mat - bridge + yw + xw - mat
    
    # --- Plot --- # 
    green  = color_cycle(3)
    orange = color_cycle(4)
    blue = color_cycle(6)
    

    # --- Inverse --- #
    br_xmin, br_ymin, br_xmax, br_ymax = plot_sheet(bridge, ax1, atom_radii, facecolor = blue, edgecolor = 'black', alpha = 0.3)
    yw_xmin, yw_ymin, yw_xmax, yw_ymax = plot_sheet(yw, ax1, atom_radii, facecolor = orange, edgecolor = 'black', alpha = 0.3)
    xw_xmin, xw_ymin, xw_xmax, xw_ymax = plot_sheet(xw, ax1, atom_radii, facecolor = orange, edgecolor = 'black', alpha = 0.3)
    plot_sheet(remaning, ax1, atom_radii, facecolor = shade_color(green, 1.5), edgecolor = 'black')
    # br_xmin, br_ymin, br_xmax, br_ymax = plot_sheet(bridge, ax1, atom_radii, facecolor = green, edgecolor = 'black', alpha = 0.3)
    # yw_xmin, yw_ymin, yw_xmax, yw_ymax = plot_sheet(yw, ax1, atom_radii, facecolor = orange, edgecolor = 'black', alpha = 0.3)
    # xw_xmin, xw_ymin, xw_xmax, xw_ymax = plot_sheet(xw, ax1, atom_radii, facecolor = blue, edgecolor = 'black', alpha = 0.3)
    # plot_sheet(remaning, ax1, atom_radii, facecolor = shade_color(color_cycle(1), 1.5), edgecolor = 'black')


    # Annotate
    arrowprops = {'arrowstyle': '<->', 'color': 'black', 'lw': 1.5}
    bbox = dict(facecolor='white', edgecolor = 'None',  alpha=0.8)
    # xw
    ax1.text((xw_xmax+xw_xmin)/2, xw_ymin - 5, 'x-width', horizontalalignment = 'center', bbox=bbox)
    ax1.annotate('', xy=(xw_xmin - 2*atom_radii, xw_ymin - 2), xytext=(xw_xmax + 2*atom_radii, xw_ymin - 2), textcoords='data', arrowprops=arrowprops)
    
    # yw
    ax1.text(br_xmin - 7, (yw_ymin+yw_ymax)/2, 'y-width', horizontalalignment = 'right', bbox=bbox)
    ax1.annotate('', xy=(br_xmin - 6, yw_ymin - 2*atom_radii), xytext=(br_xmin - 6, yw_ymax + 2*atom_radii), textcoords='data', arrowprops=arrowprops)
    
    # bridge thickness
    ax1.text((br_xmax+br_xmin)/2 + 5, br_ymin - 5, 'bridge thickness', horizontalalignment = 'right', bbox=bbox) 
    ax1.annotate('', xy=(br_xmin - 2*atom_radii, br_ymin - 2), xytext=(br_xmax + 2*atom_radii, br_ymin - 2), textcoords='data', arrowprops=arrowprops)
    
    # bridge length
    ax1.text(br_xmin - 7, (br_ymin+br_ymax)/2 - (bridge_len//4+1)*(br_ymax-br_ymin)/bridge_len , 'bridge length', horizontalalignment = 'right', bbox=bbox)
    ax1.annotate('', xy=(br_xmin - 6, br_ymin - 2*atom_radii), xytext=(br_xmin - 6, br_ymax + 2*atom_radii), textcoords='data', arrowprops=arrowprops)
    
    
    
    # --- Pattern --- #
    plot_sheet(mat, ax2, atom_radii, facecolor = 'grey', edgecolor = 'black')
    
    
    # Background
    plot_sheet(mat-bridge, ax1, atom_radii, facecolor = 'None', edgecolor = 'black', alpha = 0.2)
    plot_sheet(np.ones(shape)-mat, ax2, atom_radii, facecolor = 'None', edgecolor = 'black', alpha = 0.2)

    
    # Center elements 
    plot_center_coordinates(np.shape(mat), ax1, center_radii, facecolor = blue, edgecolor = None)
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
        fig1.savefig('../article/figures/system/honeycomb_inverse.pdf', bbox_inches='tight')
        fig2.savefig('../article/figures/system/honeycomb_pattern.pdf', bbox_inches='tight')

    
    
    pass   
    
 
        
def honeycomb_flavors(save = False):
    # Settings
    shape = (40,80)
    
    patterns = [(1,1,5,5), (1,2,1,9), (3,2,3,1), (3,1,1,3)]
    
    atom_radii = 0.6
    center_radii = 0.2
    blue = color_cycle(6)
    
    fig, axes = plt.subplots(1, 4, num = unique_fignum(), figsize = (12,4))
    # fig, axes = plt.subplots(1, 4, num = unique_fignum(),  dpi=80, facecolor='w', edgecolor='k')
    
    for i, p in enumerate(patterns):
        mat = honeycomb(shape, p[0], p[1], p[2], p[3], None)
        ax = axes[i]
        name = f'{((1+p[0]//2), p[1], p[2], p[3])}'
        # ax = axes[i//axes.shape[1], i%axes.shape[1]]
        
        plot_sheet(mat, ax, atom_radii, facecolor = 'grey', edgecolor = 'black') # Pattern   
        plot_sheet(1-mat, ax, atom_radii, facecolor = 'None', edgecolor = 'black', alpha = 0.2)  # Background
        plot_center_coordinates(np.shape(mat), ax, center_radii, facecolor = blue, edgecolor = None) # Center elements
        
        # plot settings
        ax.set_title(name)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        
    
    # Set axies
    fig.supxlabel(r"$x$ (armchair direction)", fontsize = 14)
    fig.supylabel(r"$y$ (zigzag direction)", fontsize = 14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
   
    if save:
        fig.savefig('../article/figures/system/honeycomb_flavors.pdf', bbox_inches='tight')
    
    
def bias_prop_distirbution(save = False):
    

    green  = color_cycle(3)
    orange = color_cycle(4)
    blue = color_cycle(6)
    atom_radii = 0.6
    center_radii = 0.2
    
    
    # --- Directions --- #
    fig1 = plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k'); ax1 = fig1.gca()
    mat = np.ones((2,6))
    mat[:, -1] = 0
    plot_sheet(mat, ax1, atom_radii, facecolor = 'grey', edgecolor = 'black', alpha = 0.3)
    BL, TL = plot_center_coordinates(np.shape(mat), ax1, center_radii, facecolor = blue, edgecolor = None)
    
    Cdis = 1.461
    a = 3*Cdis/np.sqrt(3)
    Bx = a/(np.sqrt(3)*2)
    vecax = a*np.sqrt(3)/2
    vecay = a/2
    Lx = (vecax + Bx)/2 
    xs, ys = BL[0], BL[1] # Start
    center = (xs + (1+ 3/2*(1 - 1)) * Lx, ys + a*(1/2 + 1/2*1 - 1/2*(2%2)) )
    neigh, directions = connected_neigh_center_elem((1,1))
    directions[:2] *= np.linalg.norm(directions[2])
    
    num = np.array([3, 2, 5, 4, 1, 0])
    for i, pos in enumerate(center + directions):
            circle = plt.Circle((pos[0], pos[1]), center_radii*1.5, color = blue)
            ax1.add_patch(circle) # direct way
            eps = 0.04
            ax1.text(pos[0], pos[1]-eps, f'{num[i]}', fontsize = 15, ha = 'center', va = 'center')
    
    from_pos = center + 0.1*directions 
    to_pos = center + 0.9*directions 
    bias = np.array((3, 1))
    
    arrowprops = dict(facecolor='black', lw = 1)
    ax1.annotate('', xy=(to_pos[0]), xytext= (from_pos[0]), textcoords='data', arrowprops=arrowprops)
    ax1.annotate('', xy=(to_pos[1]), xytext= (from_pos[1]), textcoords='data', arrowprops=arrowprops)
    ax1.annotate('', xy=(to_pos[2]), xytext= (from_pos[2]), textcoords='data', arrowprops=arrowprops)
    ax1.annotate('', xy=(to_pos[3]), xytext= (from_pos[3]), textcoords='data', arrowprops=arrowprops)
    ax1.annotate('', xy=(to_pos[4]), xytext= (from_pos[4]), textcoords='data', arrowprops=arrowprops)
    ax1.annotate('', xy=(to_pos[5]), xytext= (from_pos[5]), textcoords='data', arrowprops=arrowprops)
    ax1.annotate('', xy=(center[0] + 0.9*bias[0], center[1] + 0.9*bias[1]), xytext= (center[0] + 0.1*bias[0], center[1] + 0.1*bias[1]), textcoords='data', arrowprops=dict(facecolor='orange', lw = 1))
    ax1.text(center[0] + 0.9*bias[0] - 0.4, center[1] + 0.9*bias[1] - 0.6, 'Bias', fontsize = 15)
    
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')
    ax1.set_facecolor("white")
    fig1.set_edgecolor("white")
    
    # --- Probability distribution --- #
    fig2, axes = plt.subplots(1, 2, num = unique_fignum(), dpi=80, gridspec_kw ={'width_ratios': [1, 0.05]})
    ax2, ax22 = axes
    # fig2 = plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k'); ax2 = fig2.gca()
    
    B = np.linspace(1, 10, 7)
    cmap = 'coolwarm'
    for b in B:
        line_color = get_color_value(b, np.min(B), np.max(B), scale = 'linear', cmap=cmap)
        theta_con = np.linspace(0, np.pi, int(1e3))
        cos_theta_dis = np.dot(directions, bias)/(np.linalg.norm(bias)*np.linalg.norm(directions, axis = 1))
        p_con = np.exp(b*np.cos(theta_con))
        p_dis = np.exp(b*cos_theta_dis)
        norm = np.sum(p_dis)
        

        ax2.plot(theta_con/np.pi, p_con/norm, color = line_color, zorder = -1)
        ax2.scatter(np.arccos(cos_theta_dis)/np.pi, p_dis/norm, facecolor = line_color)
        ax2.set_xlabel(r'$\theta$ [$\pi$]', fontsize = 14)
        ax2.set_ylabel(r'$p(\theta)$', fontsize = 14)
        
    angles = np.arccos(cos_theta_dis)/np.pi
    for a in angles:
        vline(ax2, a, linewidth = 1, linestyle = '--', color = 'black', alpha = 0.5)
    
    ax3 = ax2.twiny()
    ax3.set_xlim(ax2.get_xlim())
    ax3.set_xticks(angles)
    ax3.set_xticklabels(num[np.arange(len(angles))])
    ax3.set(xlabel='Direction indexes')
    ax3.xaxis.label.set_fontsize(14)
    
    
    norm = matplotlib.colors.Normalize(np.min(B), np.max(B))
    cb = fig2.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax = ax22)
    cb.set_label(label = r'Bias strength', fontsize=14)
    
    # cb = fig2.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax2)
    
    # --- Save --- #
    fig1.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    fig2.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    if save:
        fig1.savefig('../article/figures/system/bias_prob_a.pdf', bbox_inches='tight')
        fig2.savefig('../article/figures/system/bias_prob_b.pdf', bbox_inches='tight')
        

    

    
def show_min_dis(save = False):
    pass
    

if __name__ == '__main__':
    # show_pop_up(save = False)
    # pop_up_flavors(save = False)
    # show_honeycomb(save = False)
    # honeycomb_flavors(save = False)
    
    bias_prop_distirbution(save = True)
    plt.show()