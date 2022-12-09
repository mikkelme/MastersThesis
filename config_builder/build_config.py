import sys
sys.path.append('../') # parent folder: MastersThesis

from graphene_sheet.build_graphene_sheet import *
from build_substrate import *
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
        self.bottom_substrate_freeze = 5.5 # [Å]
        self.contact_depth = 8 # [Å]
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
            minmax = get_minmax(self.sheet)
            Lx, Ly = minmax[1,:2] - minmax[0,:2]
            subpos = np.array([Lx + 20, Ly*1.5 + 20, 16]) # Lx, Ly, Lz
    
            # Set substrate Lx, Ly, Lz from input (overwrite default values)
            for i, pos in enumerate(substrate):
                if pos is not None:
                    subpos[i] = pos
                    
            # Build
            self.substrate = build_substrate(subpos, self.a_Si)
        else:
            exit(f'substrate = {substrate}, is not understood.')
            
            
        self.obj_dict['substrate'] = self.substrate
        self.is_build = False
        
        
        
        
    def save(self, object = 'all', ext = '', path = '.'):
        """ Save configuration as lammps txt file """
        if not self.is_build: self.build()
            
        # Get updated minmax        
        minmax_sheet = get_minmax(self.sheet)
        minmax_substrate = get_minmax(self.substrate)
        
        # Set order for types to insure consistensy data file
        specorder = ['C', self.substrate.get_chemical_symbols()[0]]
        
        # --- Compute info --- #
        PB_len = self.PB_rows/self.mat.shape[1] * (minmax_sheet[1,1] - minmax_sheet[0,1])
        sheet_pos = self.sheet.get_positions()
        
        PB_yhi = np.max(sheet_pos[sheet_pos[:,1] <  minmax_sheet[0,1] + PB_len + self.eps, 1])
        PB_ylo = np.min(sheet_pos[sheet_pos[:,1] >  minmax_sheet[1,1] - PB_len - self.eps, 1])
        PB_zlo = (minmax_sheet[0,2] + minmax_substrate[1,2])/2
        
        PB_lim = [PB_yhi, PB_ylo, PB_zlo]
        PB_varname = ['yhi', 'ylo', 'zlo']

        substrate_freeze_zhi = minmax_substrate[0,2] + self.bottom_substrate_freeze
        substrate_contact_zlo = minmax_substrate[1,2] - self.contact_depth 

        # --- Write data --- #
        if ext != '' : ext = f'_{ext}' 
        savename = os.path.join(path, self.name_dict[object]) + ext
        
        # Lammps data
        lammpsdata.write_lammps_data(savename + '.txt', atoms = self.obj_dict[object], 
                                                        specorder = specorder, 
                                                        velocities = True)
    
        # Info file
        outfile = open(savename + '_info.in', 'w')
        
        # PB 
        for i in range(len(PB_lim)):
            outfile.write(f'variable pullblock_{PB_varname[i]} equal {PB_lim[i]}\n') 

        # Substrate
        outfile.write(f'variable substrate_freeze_zhi equal {substrate_freeze_zhi}\n') 
        outfile.write(f'variable substrate_contact_zlo equal {substrate_contact_zlo}\n') 



    def view(self, object = 'all'):
        if not self.is_build: self.build()
        obj = self.obj_dict[object]      
        assert obj is not None, f"{object}-object is not created"
        view(obj)
        


if __name__ == "__main__":
    multiples = (3, 5)  
    unitsize = (5,7)
    
    # multiples = (9,14)  
    # unitsize = (13,15)
    
    # mat = np.ones((4, 6))
    mat = pop_up_pattern(multiples, unitsize, sp = 2)
    # mat[:, :] = 1 # Nocuts
    
    
    # substrate_file = "../substrate/crystal_Si_substrate_test.txt"


    builder = config_builder(mat)
    builder.add_pullblocks()
    # builder.add_substrate(substrate_file)
    builder.add_substrate([None, None, None])
    builder.build()
    builder.view('all')
    # builder.save("all", ext = 'in_lammps', path = '.')
    
    