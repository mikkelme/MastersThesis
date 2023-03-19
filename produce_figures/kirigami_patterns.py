from indexing import *
import colorsys

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
            
    
  


def shade_color(color, offset = 1):
    rgb = matplotlib.colors.ColorConverter.to_rgb(color)
    h, l, s =  colorsys.rgb_to_hls(*rgb) # Hue, lightness, sauration
    
    new_color = (h, l*offset, s) 
    return colorsys.hls_to_rgb(*new_color)


def show_pop_up(save = False):
    # Settings
    shape = (30,50)
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

    unit1 = delete_atoms(np.ones(shape), center_elem_trans_to_atoms(del_unit1, full = True))
    unit2 = delete_atoms(np.ones(shape), center_elem_trans_to_atoms(del_unit2, full = True))
    
    # Inverse 
    unit1 = 1 - unit1
    unit2 = 1 - unit2
    
    # --- Get space indicators --- #
    up = ref[0]%2 == 0
    sp1 = 1-delete_atoms(np.ones(shape), center_elem_trans_to_atoms([ref + [size[0]//2+(i+1), (size[0]//2 + (i+2-ref[0]%2))//2] for i in range(sp+1)], full = True))
    sp2 = 1-delete_atoms(np.ones(shape), center_elem_trans_to_atoms([ref + [i+1, -(i+1) - (i+1+ref[0]%2)//2] for i in range(sp)], full = True))
    
    # --- Plot --- # 
    green  = color_cycle(3)
    blue = color_cycle(6)
    
    # Remove overlap
    sp1[1-mat == 1] = 0
    # sp2[1-mat == 1] = 0
    
    fig, axes = plt.subplots(1, 2, num = unique_fignum(), figsize = (10,5))
    plot_sheet(1-mat, axes[0], atom_radii, facecolor = shade_color(green, 2), edgecolor = 'black')
    plot_sheet(unit1, axes[0], atom_radii, facecolor = shade_color(green, 1.3), edgecolor = 'black')
    plot_sheet(mat, axes[0], atom_radii, facecolor = 'None', edgecolor = 'black', alpha = 0.2)
    plot_center_coordinates(np.shape(mat), axes[0], center_radii, facecolor = blue, edgecolor = None)
    plot_sheet(sp1, axes[0], atom_radii, facecolor = color_cycle(1), edgecolor = 'black', alpha = 0.2)
    plot_sheet(sp2, axes[0], atom_radii, facecolor = color_cycle(1), edgecolor = 'black', alpha = 0.2)
    
    plot_sheet(mat, axes[1], atom_radii, facecolor = 'grey', edgecolor = 'black')
    plot_sheet(1-mat, axes[1], atom_radii, facecolor = 'None', edgecolor = 'black', alpha = 0.2)
    plot_center_coordinates(np.shape(mat), axes[1], center_radii, facecolor = blue, edgecolor = None)
    
    
    
    axes[0].grid(False)
    axes[1].grid(False)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.show()
    
    
    exit()
    

    
    
    
    # # Inverse and build
    # unit1 = build_graphene_sheet(1 - unit1)
    # unit2 = build_graphene_sheet(1 - unit2)
    # dummy = build_graphene_sheet(np.ones(shape))
    
    # Alter color
    # unit1.set_chemical_symbols(np.array([10]*unit1.get_global_number_of_atoms()))
    # unit2.set_chemical_symbols(np.array([9]*unit2.get_global_number_of_atoms()))
    # dummy.set_chemical_symbols(np.array([6]*dummy.get_global_number_of_atoms()))
   
   
    # # Show
    # fig, axes = plt.subplots(1, 2, num = unique_fignum(), figsize = (10,5))


    # # test = unit1.get_positions()

    # # TODO: Working with plotting XXX
    
    
    # print(sheet)
    # print(unit1)
    
    
    
    
    
    
    # # plot_atoms(dummy, axes[0], radii = atom_radii, show_unit_cell = 1, scale = 1, offset = (0,0), colors = [(1, 1, 1)]*dummy.get_global_number_of_atoms())
    # plot_atoms(sheet, axes[0], radii = atom_radii, show_unit_cell = 1, scale = 1, offset = (0,0))
    # plot_atoms(unit1, axes[0], radii = atom_radii, show_unit_cell = 1, scale = 1, offset = (0,0), colors = ['limegreen']*unit1.get_global_number_of_atoms())
    # # plot_atoms(unit2, axes[0], radii = atom_radii, show_unit_cell = 1, scale = 1, offset = (0,0), colors = ['lightskyblue']*unit2.get_global_number_of_atoms())
    
    # # plot_atoms(dummy, axes[0], radii = atom_radii/10, show_unit_cell = 0, scale = 1, offset = (-(atom_radii - atom_radii/10), (-atom_radii - atom_radii/10))) # TODO: ???
    # # plot_atoms(dummy, axes[0], radii = atom_radii, show_unit_cell = 0, scale = 1, offset = (0.35,0.35))
    
    # plot_atoms(dummy, axes[1], radii = atom_radii, show_unit_cell = 0, scale = 1, offset = (0,0), colors = ['white']*dummy.get_global_number_of_atoms())
    # plot_atoms(sheet, axes[1], radii = atom_radii, show_unit_cell = 0, scale = 1, offset = (0,0))
    # # plot_atoms(dummy, axes[1], radii = atom_radii/10, show_unit_cell = 0, scale = 1, offset = (0, 0))
    # # plot_atoms(dummy, axes[1], radii = atom_radii, show_unit_cell = 0, scale = 1, offset = (0.45,0.45))
   
    
    # ax = plot_atoms(sheet, radii = atom_radii, show_unit_cell = 0, scale = 1, offset = (0,0))
    # plot_atoms(unit1, ax, radii = atom_radii, show_unit_cell = 0, scale = 1, offset = (0,0), colors = ['blue']*unit1.get_global_number_of_atoms())
    # plot_atoms(unit2, ax, radii = atom_radii, show_unit_cell = 0, scale = 1, offset = (0,0), colors = ['red']*unit2.get_global_number_of_atoms())
    
    # sheet += unit1 + unit2
    # view(sheet)
    
    
    
    ####################
    # cut = np.zeros(np.shape(mat))
    # cut[mat == 0] = 1
    
    
    #  v.custom_colors({'Mn':'green','As':'blue'})
    
    
   

if __name__ == '__main__':
    show_pop_up()