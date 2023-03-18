from indexing import *


def show_pop_up(save = False):
    shape = (60,106)
    shape = (30,60)
    # ref = np.array([shape[0]//3, shape[1]//4])
    ref = np.array([shape[0]//2, shape[1]//4])  
    sp = 2
    size = (7,5)
    mat = pop_up(shape, size, sp, ref = ref)
    sheet = build_graphene_sheet(mat)
    
    atom_radii = 0.8
    center_radii = 0.65

    
    

    
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

            unit1 = delete_atoms(unit1, center_elem_trans_to_atoms(del_map1, full = True))
            unit2 = delete_atoms(unit2, center_elem_trans_to_atoms(del_map2, full = True))
    
    

    unit1 = delete_atoms(np.ones(shape), center_elem_trans_to_atoms(del_unit1, full = True))
    unit2 = delete_atoms(np.ones(shape), center_elem_trans_to_atoms(del_unit2, full = True))
        
    # Inverse and build
    unit1 = build_graphene_sheet(1 - unit1)
    unit2 = build_graphene_sheet(1 - unit2)
    dummy = build_graphene_sheet(np.ones(shape))
    
    # Alter color
    # unit1.set_chemical_symbols(np.array([10]*unit1.get_global_number_of_atoms()))
    # unit2.set_chemical_symbols(np.array([9]*unit2.get_global_number_of_atoms()))
    # dummy.set_chemical_symbols(np.array([6]*dummy.get_global_number_of_atoms()))
   
   
    # Show
    fig, axes = plt.subplots(1, 2, num = unique_fignum(), figsize = (10,5))


    # test = unit1.get_positions()

    # TODO: Working with plotting XXX
    
    
    plot_atoms(dummy, axes[0], radii = atom_radii, show_unit_cell = 1, scale = 1, offset = (0,0), colors = [(1, 1, 1)]*dummy.get_global_number_of_atoms())
    plot_atoms(sheet, axes[1], radii = atom_radii, show_unit_cell = 0, scale = 1, offset = (0,0))
    plot_atoms(unit1, axes[0], radii = atom_radii, show_unit_cell = 1, scale = 1, offset = (0,0), colors = ['limegreen']*unit1.get_global_number_of_atoms())
    plot_atoms(unit2, axes[0], radii = atom_radii, show_unit_cell = 1, scale = 1, offset = (0,0), colors = ['lightskyblue']*unit2.get_global_number_of_atoms())
    
    # plot_atoms(dummy, axes[0], radii = atom_radii/10, show_unit_cell = 0, scale = 1, offset = (-(atom_radii - atom_radii/10), (-atom_radii - atom_radii/10))) # TODO: ???
    # plot_atoms(dummy, axes[0], radii = atom_radii, show_unit_cell = 0, scale = 1, offset = (0.35,0.35))
    
    plot_atoms(dummy, axes[1], radii = atom_radii, show_unit_cell = 0, scale = 1, offset = (0,0), colors = ['white']*dummy.get_global_number_of_atoms())
    plot_atoms(sheet, axes[1], radii = atom_radii, show_unit_cell = 0, scale = 1, offset = (0,0))
    # plot_atoms(dummy, axes[1], radii = atom_radii/10, show_unit_cell = 0, scale = 1, offset = (0, 0))
    # plot_atoms(dummy, axes[1], radii = atom_radii, show_unit_cell = 0, scale = 1, offset = (0.45,0.45))
   
   
    axes[0].grid(False)
    axes[1].grid(False)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    # ax = plot_atoms(sheet, radii = atom_radii, show_unit_cell = 0, scale = 1, offset = (0,0))
    # plot_atoms(unit1, ax, radii = atom_radii, show_unit_cell = 0, scale = 1, offset = (0,0), colors = ['blue']*unit1.get_global_number_of_atoms())
    # plot_atoms(unit2, ax, radii = atom_radii, show_unit_cell = 0, scale = 1, offset = (0,0), colors = ['red']*unit2.get_global_number_of_atoms())
    
    plt.show()
    # sheet += unit1 + unit2
    # view(sheet)
    
    
    
    ####################
    # cut = np.zeros(np.shape(mat))
    # cut[mat == 0] = 1
    
    
    #  v.custom_colors({'Mn':'green','As':'blue'})
    
    
   

if __name__ == '__main__':
    show_pop_up()