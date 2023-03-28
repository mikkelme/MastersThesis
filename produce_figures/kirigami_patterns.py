from indexing import *
from stretch_profiles import *
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
            if ax is not None:
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
    
    

def RW_flavors(save = False):
    # Settings
    size = (62,106)
    np.random.seed(0)
    
    direc = {'up': (0, 1), 
             'down': (0, -1),
             'up_right': (np.tan(np.pi/3), 1), 
             'up_left': (-np.tan(np.pi/3), 1),
             'down_right': (np.tan(np.pi/3), -1),
             'down_left': (-np.tan(np.pi/3), -1)}

    
    atom_radii = 0.6
    center_radii = 0.2
    blue = color_cycle(6)
    
    
    # --- Patterns --- #
    SET = []
    # Order
    SET += [RW_Generator(size, num_walks = 25, max_steps = 15, min_dis = 0, bias = [direc['up_right'], 100], center_elem = False,  avoid_unvalid = False,  RN6 = False,  grid_start = True,  centering = True,  stay_or_break = 0,  avoid_clustering = 10,  periodic = True)]
    SET += [RW_Generator(size, num_walks = 25, max_steps = 15, min_dis = 0, bias = [direc['up_right'], 100], center_elem = False,  avoid_unvalid = False,  RN6 = True,  grid_start = True,  centering = True,  stay_or_break = 0,  avoid_clustering = 10,  periodic = True)]
    
    # Stay or break
    SET += [RW_Generator(size, num_walks = 20, max_steps = 30, min_dis = 4, bias = [(0,0), 0], center_elem = False,  avoid_unvalid = True,  RN6 = True,  grid_start = False,  centering = False,  stay_or_break = 0.9,  avoid_clustering = 10,  periodic = True)]
    
    # Traditional 
    SET += [RW_Generator(size, num_walks = 30, max_steps = 40, min_dis = 4, bias = [(0,0), 0], center_elem = False,  avoid_unvalid = True,  RN6 = False,  grid_start = False,  centering = False,  stay_or_break = 0,  avoid_clustering = 10,  periodic = True)]
    
    # Slight bias
    SET += [RW_Generator(size, num_walks = 20, max_steps = 30, min_dis = 4, bias = [direc['down_right'], 1.2], center_elem = False,  avoid_unvalid = True,  RN6 = False,  grid_start = False,  centering = False,  stay_or_break = 0,  avoid_clustering = 10,  periodic = True)]
    
    # High porosity
    SET += [RW_Generator(size, num_walks = 32, max_steps = 30, min_dis = 4, bias = [direc['down_left'], 1.2], center_elem = 'full', avoid_unvalid = True,  RN6 = False,  grid_start = False,  centering = False,  stay_or_break = 0,  avoid_clustering = 10,  periodic = True)]
    
    
    fig, axes = plt.subplots(2, 3, num = unique_fignum(), figsize = (12,8))
    names = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)', '(o)', '(p)']
    for i, set in enumerate(SET):
        print(i)
        mat = set.generate()
        ax = axes[(i)//axes.shape[1], (i)%axes.shape[1]]
        # builder = config_builder(mat)
        # builder.view()
        
        plot_sheet(mat, ax, atom_radii, facecolor = 'grey', edgecolor = 'black') # Pattern   
        plot_sheet(1-mat, ax, atom_radii, facecolor = 'None', edgecolor = 'black', alpha = 0.2)  # Background
        plot_center_coordinates(np.shape(mat), ax, center_radii, facecolor = blue, edgecolor = None) # Center elements
        
        # plot settings
        ax.set_title(names[i], y=-0.05)
        # ax.set_xlabel(names[i], fontsize = 14)
        ax.grid(False)
        ax.set_facecolor("white")
        ax.axis('off')
        
    # Set axies
    fig.supxlabel(r"$x$ (armchair direction)", fontsize = 14)
    fig.supylabel(r"$y$ (zigzag direction)", fontsize = 14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
   
    if save:
        fig.savefig('../article/figures/system/RW_flavors.pdf', bbox_inches='tight')
    
    
    
def bias_prop_distribution(save = False):
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
    
    # num = np.array([3, 2, 5, 4, 1, 0])
    # num = np.array([2, 3, 0, 1, 4, 5])
    num = np.array([3, 4, 1, 2, 5, 6])
    for i, pos in enumerate(center + directions):
            circle = plt.Circle((pos[0], pos[1]), center_radii*1.5, color = blue)
            ax1.add_patch(circle) # direct way
            eps = 0.04
            ax1.text(pos[0], pos[1]-eps, f'{num[i]}', fontsize = 15, ha = 'center', va = 'center')
    
    from_pos = center + 0.1*directions 
    to_pos = center + 0.9*directions 
    bias = np.array((3, 1))
    
    arrowprops = dict(facecolor='black', lw = 1)
    for i in range(len(from_pos)):
        ax1.annotate('', xy=(to_pos[i]), xytext= (from_pos[i]), textcoords='data', arrowprops=arrowprops)
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
    
    B = np.logspace(-1, 1.3, 8)
    cmap = 'coolwarm'
    for i, b in enumerate(B):
        line_color = get_color_value(b, np.min(B), np.max(B), scale = 'log', cmap=cmap)
        theta_con = np.linspace(0, np.pi, int(1e3))
        cos_theta_dis = np.dot(directions, bias)/(np.linalg.norm(bias)*np.linalg.norm(directions, axis = 1))
        p_con = np.exp(b*np.cos(theta_con))
        p_dis = np.exp(b*cos_theta_dis)
        norm = np.sum(p_dis)
        

        ax2.plot(theta_con/np.pi, p_con/norm, color = line_color, zorder = i)
        ax2.scatter(np.arccos(cos_theta_dis)/np.pi, p_dis/norm, facecolor = line_color, edgecolor = 'black', zorder = i)
        ax2.set_xlabel(r'$\theta$ [$\pi$]', fontsize = 14)
        ax2.set_ylabel(r'$p(\theta)$', fontsize = 14)
        
    angles = np.arccos(cos_theta_dis)/np.pi
    # for a in angles:
    #     vline(ax2, a, linewidth = 1, linestyle = '--', color = 'black', alpha = 0.5)
    
    ax3 = ax2.twiny()
    ax3.set_xlim(ax2.get_xlim())
    ax3.set_xticks(angles)
    ax3.set_xticklabels(num[np.arange(len(angles))])
    ax3.grid(False)
    ax3.set(xlabel='Direction indexes')
    ax3.xaxis.label.set_fontsize(14)
    
    
    # norm = matplotlib.colors.Normalize(np.min(B), np.max(B))
    norm = matplotlib.colors.LogNorm(np.min(B), np.max(B))
    
    cb = fig2.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax = ax22)
    cb.set_label(label = r'Bias strength', fontsize=14)
    
    
    # --- Save --- #
    fig1.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    fig2.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    
    if save:
        fig1.savefig('../article/figures/system/bias_prob_a.pdf', bbox_inches='tight')
        fig2.savefig('../article/figures/system/bias_prob_b.pdf', bbox_inches='tight')
        

def stay_or_break(save = False):
    green  = color_cycle(3)
    orange = color_cycle(4)
    blue = color_cycle(6)
    atom_radii = 0.6
    center_radii = 0.2
    
    
    # --- Directions --- #
    fig1 = plt.figure(num=unique_fignum(), dpi=80, facecolor='w', edgecolor='k'); ax1 = fig1.gca()
    mat = np.ones((4,6))
    mat[:, 0] = 0
    mat[0, [1,2,4,5]] = 0
    mat[-1, [1,2,4,5]] = 0
    plot_sheet(mat, ax1, atom_radii, facecolor = 'grey', edgecolor = 'black', alpha = 0.3)
    BL, TL = plot_center_coordinates(np.shape(mat), None, center_radii, facecolor = blue, edgecolor = None)
    
    Cdis = 1.461
    a = 3*Cdis/np.sqrt(3)
    Bx = a/(np.sqrt(3)*2)
    vecax = a*np.sqrt(3)/2
    vecay = a/2
    Lx = (vecax + Bx)/2 
    xs, ys = BL[0], BL[1] # Start
    i, j = 2, 2
    center = (xs + (1+ 3/2*(i - 1)) * Lx, ys + a*(1/2 + 1/2*j - 1/2*(i%2)) )
    neigh, directions = connected_neigh_center_elem((1,1))
    directions[:2] *= np.linalg.norm(directions[2])
    
    # Plot center elements
    circle = plt.Circle((center[0], center[1]), center_radii, color = blue)
    ax1.add_patch(circle) # direct way
    for i, pos in enumerate(center + directions):
            circle = plt.Circle((pos[0], pos[1]), center_radii, color = blue)
            ax1.add_patch(circle) # direct way
    
    
    # Directions
    from_center = center + 0.1*directions 
    to_center = center + 0.9*directions 
    arrowprops_center = dict(facecolor='black', lw = 1)
    
    arrowprops_atom_even = dict(facecolor=color_cycle(1), width = 2, headwidth = 12, edgecolor = 'None')
    arrowprops_atom_odd = dict(facecolor=color_cycle(4), width = 2, headwidth = 12, edgecolor = 'None')
    atom_even = np.array([[-2/np.sqrt(3), 0], [1/np.sqrt(3), 1], [1/np.sqrt(3), -1]])*a/2
    atom_odd = np.array([[2/np.sqrt(3), 0], [-1/np.sqrt(3), 1], [-1/np.sqrt(3), -1]])*a/2
    
    
    from_atom_even = center + atom_even
    to_atom_even = center + 2*atom_even
    from_atom_odd = center + atom_odd
    to_atom_odd = center + 2*atom_odd
    
    
    # Center directions
    for i in range(len(from_center)):
        ax1.annotate('', xy=(to_center[i]), xytext= (from_center[i]), textcoords='data', arrowprops=arrowprops_center) 
        
    # Atom directions
    for i in range(len(from_atom_even)):
        an_even = ax1.annotate('', xy=(to_atom_even[i]), xytext= (from_atom_even[i]), textcoords='data', arrowprops=arrowprops_atom_even, label = 'Even') 
        an_odd = ax1.annotate('', xy=(to_atom_odd[i]), xytext= (from_atom_odd[i]), textcoords='data', arrowprops=arrowprops_atom_odd, label = 'Odd') 
    
    
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')
    ax1.set_facecolor("white")
    fig1.set_edgecolor("white")
    fig1.legend([an_even.arrow_patch, an_odd.arrow_patch], (an_even.get_label(), an_odd.get_label()), fontsize = 14)
    
    # --- Save --- #
    fig1.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    if save:
        fig1.savefig('../article/figures/system/stay_or_break.pdf', bbox_inches='tight')
        

def grid_start(save = False):
    mat = np.ones((14, 18))
    atom_radii = 0.6
    center_radii = 0.2
    blue = color_cycle(6)
    cmap = 'terrain'
    cmap = plt.get_cmap('terrain')
    cmap = truncate_colormap(cmap, 0, 0.85)
    np.random.seed(1)
    
    fig, axes = plt.subplots(3, 3, num = unique_fignum(), figsize = (12,7))
    for nw in range(1, 10):
        print(nw)
        ax = axes[(nw-1)//axes.shape[1], (nw-1)%axes.shape[1]]
        mat[:, :] = 1
        
        # Update grid to nearest square size (works better visually )
        if np.sqrt(nw-1)%1 < 0.01:
            num_walks = np.ceil(np.sqrt(nw))**2
            RW = RW_Generator(size = np.shape(mat), num_walks = num_walks, grid_start = True, center_elem = False)
            RW.initialize()
            grid = RW.get_grid()
        
        
        for i, g in enumerate(grid[:nw]):
            g_mat = np.zeros(np.shape(mat))
            g_mat[g[0], g[1]] = 1
            mat[g[0], g[1]] = 0
            color = get_color_value(i+1, 1, 9, scale = 'linear', cmap=cmap)
            plot_sheet(g_mat, ax, atom_radii, facecolor = color, edgecolor = 'black') # Pattern   
            
        plot_sheet(mat, ax, atom_radii, facecolor = 'None', edgecolor = 'black', alpha = 0.5)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
            
    fig.supxlabel(r"$x$ (armchair direction)", fontsize = 14)
    fig.supylabel(r"$y$ (zigzag direction)", fontsize = 14)
    fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2,)

    norm = matplotlib.colors.Normalize(0.5, 9.5)
    bounds = np.linspace(0.5, 9.5, 10)
    cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), spacing='proportional', boundaries=bounds, ticks=np.arange(1, 10), ax=axes.ravel().tolist())
    cb.set_label(label = r'Ordering', fontsize=14)
    
    
    # cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=np.arange(1, 10), ax=axes.ravel().tolist())
    # divider = make_axes_locatable(axes)
    # cax = divider.append_axes("right", "5%", pad="3%")
    # cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=np.arange(1, 10), cax=cax)
   
    if save:
        fig.savefig('../article/figures/system/grid_start.pdf', bbox_inches='tight')
    
  

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
    

def show_all_conf():
    root = '../config_builder/'
    data_root = [root+'popup', root+'honeycomb', root+'RW']
    atom_radii = 0.6
    
    for path in data_root:
        if path == data_root[0]: print('Skip Popup') ;continue
        if path == data_root[1]: print('Skip Honeycomb'); continue
        if path == data_root[2]: print('Skip RW'); continue
        files = np.sort(get_files_in_folder(path, ext = '.npy'))
        shape = (6, 5)
        num_files = len(files)
        num_axes = shape[0]*shape[1]
        
        figs = [plt.subplots(shape[0], shape[1], num = unique_fignum(), figsize = (shape[1],shape[0])) for i in range(num_files//num_axes)]
        axes_left = num_files - num_axes*len(figs)
        last_shape = (int(np.ceil(axes_left/shape[0])), 5)
        total_num_axes = num_axes*len(figs) + last_shape[0]*last_shape[1]
        figs.append(plt.subplots(last_shape[0], last_shape[1], num = unique_fignum(), figsize = (last_shape[1],last_shape[0])))
        
        for i in range(total_num_axes):
            if i%num_axes == 0:
                fig, axes = figs[i//num_axes]
                offset = num_axes*(i//num_axes)
                
            # if i < int(np.floor(num_files/num_axes)*num_axes):
            #     continue # Only go to last fig
            
            if i < int((np.floor(num_files/num_axes)-1)*num_axes):
                continue # Only go to last fig
            
            idx = (i - offset)//shape[1], (i - offset)%shape[1]
            ax = axes[idx]
            print(i, idx)
            
            if i < num_files:
                file = files[i]
                # if i%6 == 0: # TMP
                mat = np.load(file)
                plot_sheet(mat, ax, atom_radii, facecolor = 'black', edgecolor = 'black', linewidth = 0.1, alpha = 1)
                name, pattern_type = get_name(path, file)
                ax.set_title(name, pad = -0.1, fontsize = 5)
                
            ax.axis('equal')
            ax.axis('off')
            ax.axis('equal')
                
            # ax.grid(False)
            # ax.set_xticks([])
            # ax.set_yticks([])
            # ax.set_facecolor("white")
            
        for i, fig in enumerate(figs):
            fig[0].supxlabel(r"$x$ (armchair direction)", fontsize = 14)
            fig[0].supylabel(r"$y$ (zigzag direction)", fontsize = 14)
            if i == len(figs)-1:
                f = figs[len(figs)-2][0] # XXX
                left = f.subplotpars.left
                bottom = f.subplotpars.bottom
                right = f.subplotpars.right
                top = f.subplotpars.top
                wspace = f.subplotpars.wspace
                hspace = f.subplotpars.hspace
                print(left, bottom, right, top, wspace, hspace)
                fig[0].subplots_adjust(left, bottom, right, top, wspace, hspace)
            else:
                fig[0].tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2,)
            
            fig[0].savefig(f'../article/figures/dataset/{path.split("/")[-1]}_{i}_FIX.pdf', bbox_inches='tight')
        # plt.show()
    
        # exit()
    
    pass


if __name__ == '__main__':
    # show_pop_up(save = False)
    # pop_up_flavors(save = False)
    # show_honeycomb(save = False)
    # honeycomb_flavors(save = False)
    
    # bias_prop_distribution(save = False)
    # stay_or_break(save = False)
    # grid_start(save = False)
    # RW_flavors(save = True)
    
    # show_all_conf()
    pass
    # plt.show()