import sys
sys.path.append('../') # parent folder: MastersThesis

from graphene_sheet.build_graphene_sheet import *
from config_builder.build_substrate import *
# from graphene_sheet.build_utils import *


class config_builder:
    def __init__(self, mat):
        self.mat = mat


        # ASE objects
        self.sheet      = None # Graphene sheet
        self.substrate  = None # Si substrate
        self.merge      = None # Sheet + substrate
    
        self.PB_rows = 0
        self.a_Si = 5.430953
    
        # Parameters
        self.Cdis = 1.461 # carbon-carbon distance [Å] TODO: Take as input from data_generator???
        self.sheet_substrate_dis = 2.8 # [Å] 
        self.bottom_substrate_freeze = 4 # [Å]
        self.contact_depth = 4 # [Å]
        self.substrate_atomic_num = 14 # Si [atomic number]
        self.eps = 1e-6
        
        
        # Dictionaries
        self.obj_dict  = {'sheet'     : self.sheet,
                          'substrate' : self.substrate,
                          'all'       : self.merge}
        
        self.name_dict = {'sheet'     : 'sheet',
                          'substrate' : 'substrate',
                          'all'       : 'sheet_substrate'}
        
        
        self.build()
        self.is_build = True
        



    def build(self):
        """ Build configuration """
        # self.build_graphene_sheet()
        self.sheet = build_graphene_sheet(self.mat, self.Cdis)
        
        
        if self.substrate is not None:
            # --- Translate sheet relatively to substrate --- #
            # Find min and max positions
            minmax_sheet = get_minmax(self.sheet)
            minmax_substrate = get_minmax(self.substrate)
            
            # Find centers
            center_sheet = (minmax_sheet[0] + minmax_sheet[1])/2
            center_substrate = (minmax_substrate[0] + minmax_substrate[1])/2

            # Align center and get right distance
            trans_vec1 = center_substrate - center_sheet
            trans_vec1[2] = self.sheet_substrate_dis - (minmax_sheet[0, 2] - minmax_substrate[1,2])
            self.sheet.translate(trans_vec1)
            minmax_sheet += trans_vec1
    
    
            # --- Merge into same object --- #
            self.merge = self.sheet + self.substrate
    
            # Fix cell/simulation box
            self.align_and_adjust_cell(self.merge)
            
            
        else:
            self.align_and_adjust_cell(self.sheet)
            
        # Update object dictionary
        self.obj_dict['sheet'] = self.sheet
        self.obj_dict['substrate'] = self.substrate
        self.obj_dict['all'] = self.merge
        self.is_build = True
    

 
    def align_and_adjust_cell(self, object):
        """ Align with origo and adjust cell """
        minmax = get_minmax(object)
        
        # Add space for PBC for substrate
        # if self.a_Si is not None: # <--------- Not valid anymore XXX XXX XXX 
        if self.substrate is not None: 
            minmax_substrate = get_minmax(self.substrate)
            minmax[1,:2] = np.max((minmax[1,:2], minmax_substrate[1,:2] + self.a_Si/4), axis = 0)
        
        trans = -minmax[0, :] + np.ones(3)*self.eps
        for obj in [self.sheet, self.substrate, self.merge]:
            if obj is not None:
                obj.translate(trans)
                obj.set_cell(minmax[1,:] + trans + np.ones(3)*self.eps)

        
    
    def add_pullblocks(self, PB_rows = 6):
        """ Add pullblocks to sheet with length PB_len """
        self.PB_rows = PB_rows
        self.mat, self.PB = build_pull_blocks(self.mat, pullblock = PB_rows)
        self.is_build = False
        
        
         
    def add_substrate(self, substrate = None):
        """ Add substrate by reading file or creating it in ASE. 
            when <substrate> is type:
            string:             Read as file.
            list/tuple/array:   Build crystal in ASE.    """
            

        if isinstance(substrate, str): # Read substrate file
            self.substrate = lammpsdata.read_lammps_data(substrate, Z_of_type=None, style='atomic', sort_by_id=True, units='metal')
            self.substrate.set_atomic_numbers(np.ones(self.substrate.get_global_number_of_atoms())*self.substrate_atomic_num) # For visualization
        
        elif hasattr(substrate, "__len__"): # Create Si crystal in ASE
            if not self.is_build: self.build()
            
            # Get default values
            Lx, Ly = self.get_sheet_size()
            subpos = np.array([Lx + 20, Ly*1.5 + 20, 16]) # Lx, Ly, Lz
            # Set substrate Lx, Ly, Lz from input (overwrite default values)
            for i, pos in enumerate(substrate):
                if pos is not None:
                    subpos[i] = pos
            subpos[:2] += self.a_Si/4 
            
            # Build
            self.substrate = build_substrate(subpos, self.a_Si)
        else:
            exit(f'substrate = {substrate}, is not understood.')
            
            
        self.obj_dict['substrate'] = self.substrate
        self.is_build = False
        
        
    def get_sheet_size(self):
        minmax = get_minmax(self.sheet)
        Lx, Ly = minmax[1,:2] - minmax[0,:2]
        return Lx, Ly
        
    def save_lammps(self, object = 'all', ext = '', path = '.'):
        """ Save configuration as lammps txt file """
        if not self.is_build: self.build()
            
        # Get updated minmax        
        minmax_sheet = get_minmax(self.sheet)
        
        # Set order for types to insure consistensy data file
        # specorder = ['C', self.substrate.get_chemical_symbols()[0]]
        specorder = ['C', 'Si']
        
        # --- Compute info --- #
        PB_len = self.PB_rows/self.mat.shape[1] * (minmax_sheet[1,1] - minmax_sheet[0,1])
        sheet_pos = self.sheet.get_positions()
        
        PB_yhi = np.max(sheet_pos[sheet_pos[:,1] <  minmax_sheet[0,1] + PB_len + self.eps, 1])
        PB_ylo = np.min(sheet_pos[sheet_pos[:,1] >  minmax_sheet[1,1] - PB_len - self.eps, 1])
        
        PB_lim = [PB_yhi, PB_ylo]
        PB_varname = ['yhi', 'ylo']


        # --- Write data --- #
        if ext != '' : ext = f'_{ext}' 
        savename = os.path.join(path, self.name_dict[object]) + ext
        config_file = savename + '.txt'
        info_file = savename + '_info.in'
        
        
        # Lammps data
        lammpsdata.write_lammps_data(config_file, atoms = self.obj_dict[object], 
                                                        specorder = specorder, 
                                                        velocities = True)
    
        # Info file
        outfile = open(info_file, 'w')
        
        # PB 
        for i in range(len(PB_lim)):
            outfile.write(f'variable pullblock_{PB_varname[i]} equal {PB_lim[i]}\n') 

        # Substrate
        if self.substrate is not None:
            outfile.write(f'variable substrate_freeze_depth equal {self.bottom_substrate_freeze}\n') 
            outfile.write(f'variable substrate_contact_depth equal {self.contact_depth}\n')


        return config_file, info_file

    def save_mat(self, path, prefix = 'conf'):
        file_id = 1
        savename = f"{prefix}{file_id}"
        try:
            existing_data = (',').join(os.listdir(path))
        except FileNotFoundError:
            os.makedirs(path)
            existing_data = (',').join(os.listdir(path))
            
        while savename in existing_data:
            file_id += 1
            savename = f"{prefix}{file_id}"
        
        # Save matrix as array
        array_file = os.path.join(path, savename)
        np.save(array_file, self.mat)
        return array_file


    def save_view(self, object = None, path = '.', ):
        obj = self.get_object(object)
        png_file = os.path.join(path,'config.png')
        write(png_file, obj)
        return png_file

    def view(self, object = None):
        obj = self.get_object(object)
        view(obj)
        

    def get_object(self, object = None):
        if not self.is_build: self.build()
        try:
            if object is None:
                obj = self.obj_dict['all']
                if obj is None:
                    obj = self.obj_dict['sheet']     
            else:  
                obj = self.obj_dict[object] 
        except KeyError:
            exit(f'Object: \"{object}\", is not understood.')     
        
        assert obj is not None, f"\"{object}\"-object is not created"
        return obj
    
    
    
if __name__ == "__main__":
    # multiples = (8, 15) # 174x189
    # multiples = (7, 13) # 152x163
    # multiples = (6, 11) # 130x138
    # multiples = (5, 9)  # 108x113
    multiples = (4, 7)  # 86x87
    # multiples = (3, 5)  # 64x62
    # multiples = (2, 2)    # 42x24
    unitsize = (5,7)
    
    
    mat = pop_up_pattern(multiples, unitsize, sp = 2)
    # mat[:, :] = 1 # Nocuts
    

    builder = config_builder(mat)
    size_name = 'x'.join([str(round(v)) for v in builder.get_sheet_size()])
    builder.add_pullblocks()
    builder.view('sheet')
    # builder.save_mat('./cut_nocut', 'nocut')  
    builder.save("sheet", ext = f"cut_{size_name}", path = '.')
    print(size_name)
  
  
    # builder.add_substrate(substrate_file)
    # builder.add_substrate([None, None, None])
    
    