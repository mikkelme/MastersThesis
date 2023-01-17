import sys
sys.path.append('../') # parent folder: MastersThesis

from graphene_sheet.build_graphene_sheet import *
from config_builder.build_substrate import *
# from graphene_sheet.build_utils import *

from datetime import date


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

        
    def add_pullblocks(self, PB_rows = 12):
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
        
        
        # Lammps 'atomic' style data file
        tmp_data = './tmp_lammps_data.txt'
        lammpsdata.write_lammps_data(tmp_data, atoms = self.obj_dict[object], 
                                                        specorder = specorder, 
                                                        velocities = False,
                                                        units = "metal")
        

        # Rewrite to 'bond' style format
        infile = open(tmp_data, 'r')
        outfile = open(config_file, 'w')
        
        while True:
            line = infile.readline()
            outfile.write(line)
            if line[:5] == "Atoms":
                outfile.write(infile.readline())
                break
        
        for line in infile:
            words = line.split()
            words.insert(1, '0')
            words += ['0', '0', '0\n']
            outfile.write(' '.join(s for s in words))
            
        os.remove(tmp_data)
        
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

    def check_name(self, path, savename, extension, overwrite = False):
        try:
            existing_data = (',').join(os.listdir(path))
        except FileNotFoundError:
            os.makedirs(path)
            existing_data = (',').join(os.listdir(path))
            
        if overwrite: return savename + extension
        
        file_id = 0
        tryname = savename + extension
        while tryname in existing_data:
            file_id += 1
            tryname = f"{savename}_{file_id}{extension}"
            
        if file_id > 0:
            savename = tryname
        else:
            savename += extension
                    
        return savename
    
    def save_mat(self, path, savename, overwrite = True):
        savename = self.check_name(path, savename, '.npy', overwrite)
        array_file = os.path.join(path, savename)
        np.save(array_file, self.mat)
        return array_file


    def save_view(self, path = '.', object = None, savename = 'config', overwrite = True):
        savename = self.check_name(path, savename, '.png', overwrite)
        obj = self.get_object(object)
        png_file = os.path.join(path,savename)
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
    
    
    def __str__(self):
        string = 'Build status: '
        if self.is_build:
            string += 'Build\n'
        else:
            string += 'Not build\n'
        
        
        if self.is_build:
            string += 'Active object: '
            for key in self.obj_dict:
                if self.obj_dict[key] is not None:
                    string += f'{self.name_dict[key]}, '
               

            string += '\n'
        
        Lx, Ly = self.get_sheet_size()
        string += f'Sheet size: ({Lx:g}, {Ly:g})'
    

        return string
    
    
    
def pop_up_dataset(shape = (62, 106), min_sp = 1, max_sp = 4, max_cut = (9,13)):
    # Parameters
    dir = './pop_up'
    overwrite = True
    store = True
    ref = 'RAND'
    
    # Produce and store data points
    count = 0
    for sp in range(min_sp, max_sp + 1):
        for i in range(1, max_cut[0] + 1):
            if i%2 == 1:
                for j in range(1, max_cut[1] + 1):
                    if (np.abs(i - j) - 2)%4 == 0:
                        count += 1  
                        mat = pop_up(shape, size = (i,j), sp = sp, ref = ref)
                        name = f'sp{sp}({i},{j})'
                        if store:
                            print(f'\rStoring ({count:03d})| overwrite = {overwrite} | sp = {sp}/{max_sp}, size = ({i:02d},{j:02d})/({max_cut[0]},{max_cut[1]}) ', end = "")
                            builder = config_builder(mat)
                            builder.save_mat(dir, name, overwrite)
                            builder.save_view(dir, 'sheet', name, overwrite)
                        else:
                            print(f'\rCounting ({count:03d}) | sp = {sp}/{max_sp}, size = ({i:02d},{j:02d})/({max_cut[0]},{max_cut[1]}) ', end = "")

    # Write dataset info
    if store:
        with open(os.path.join(dir,'dataset_info.txt'), 'w') as outfile:
            outfile.write('DATASET INFO\n')
            outfile.write(f'Date: {date.today()}\n')
            outfile.write(f'Generated as: pop_up_dataset(shape = ({shape[0]},{shape[1]}), min_sp = {min_sp}, max_sp = {max_sp}, max_cut = ({max_cut[0]},{max_cut[1]}))\n')
            outfile.write(f'ref: {ref}\n')
            outfile.write(f'Total configurations: {count}')
        
    
def honeycomb_dataset(shape = (62, 106), min_val = (2, 1, 1, 1), max_val = (3, 5, 5, 5)):
    # max_val: {xwidth, ywidth, bridge_thickness, bridge_len}
    
    # Parameters
    dir = './honeycomb'
    overwrite = True
    store = True
    ref = 'RAND'
    

    # # Produce and store data points
    count = 0
    for xwidth in (x for x in range(min_val[0], max_val[0]+1) if x%2 == 1):
         for ywidth in range(min_val[1], max_val[1]+1):
            for bridge_thickness in (x for x in range(min_val[2], max_val[2]+1) if x%2 == 1):
                for bridge_len in (x for x in range(min_val[3], max_val[3]+1) if x%2 == 1):
                        count += 1  
                        mat = honeycomb(shape, xwidth, ywidth, bridge_thickness, bridge_len, ref = ref)
                        name = f'{xwidth}{ywidth}{bridge_thickness}{bridge_len}'
                        
                        if store:
                            print(f'\rStoring ({count:03d})| overwrite = {overwrite} | ({xwidth}, {ywidth}, {bridge_thickness}, {bridge_len})/({max_val[0]}, {max_val[1]}, {max_val[2]}, {max_val[3]})', end = "")
                            builder = config_builder(mat)
                            builder.save_mat(dir, name, overwrite)
                            builder.save_view(dir, 'sheet', name, overwrite)
                        
                        else:
                            print(f'\rCounting ({count:03d})| ({xwidth}, {ywidth}, {bridge_thickness}, {bridge_len})/({max_val[0]}, {max_val[1]}, {max_val[2]}, {max_val[3]}) ', end = "")

      # Write dataset info
    if store:
        with open(os.path.join(dir,'dataset_info.txt'), 'w') as outfile:
            outfile.write('DATASET INFO\n')
            outfile.write(f'Date: {date.today()}\n')
            outfile.write(f'Generated as: honeycomb_dataset(shape = ({shape[0]},{shape[1]}), min_val = ({min_val[0]}, {min_val[1]}, {min_val[2]}, {min_val[3]}), max_val = ({max_val[0]}, {max_val[1]}, {max_val[2]}, {max_val[3]}))\n')
            outfile.write(f'ref: {ref}\n')
            outfile.write(f'Total configurations: {count}')
        
                        

def baseline_dataset(shape = (62, 106)):
    dir = './baseline'
    overwrite = True
    ref = None
    
    # Nocut
    names =    ['nocut', 'sp1(7,5)', '3215']
    matrices = [np.ones((shape[0], shape[1])).astype('int'),
                pop_up(shape, (7,5), 1, ref),
                honeycomb(shape, 3, 2, 1, 5, ref)]
    
    for name, mat in zip(names, matrices):
        print(f'\rStoring | overwrite = {overwrite} | {name}     ', end = "")
        builder = config_builder(mat)
        builder.save_mat(dir, name, overwrite)
        builder.save_view(dir, 'sheet', name, overwrite)
    print()

    with open(os.path.join(dir,'dataset_info.txt'), 'w') as outfile:
        outfile.write('DATASET INFO\n')
        outfile.write(f'Date: {date.today()}\n')
        outfile.write(f'Generated as:\n')
        outfile.write(f'np.ones(({shape[0]}, {shape[1]})).astype(\'int\')\n')
        outfile.write(f'pop_up({shape}, (7,5), 1, {ref})\n')
        outfile.write(f'honeycomb({shape}, 3, 2, 1, 5, {ref})\n')
        outfile.write(f'Total configurations: 3')
    
    
    
  
if __name__ == "__main__":
    
    pop_up_dataset(shape = (62, 106), min_sp = 1, max_sp = 4, max_cut = (9,13))
    honeycomb_dataset(shape = (62, 106), min_val = (2, 1, 1, 1), max_val = (3, 5, 5, 5))
    baseline_dataset()
    
    pass
    
    
    # mat = honeycomb((60, 106), 3, 2, 1, 5)
    # mat = pop_up((60, 106), (5,7), 1)
    # mat = honeycomb((60, 106), 3, 2, 1, 5)
    # mat[:] = 1
    # mat[mat == 1] = 2
    # mat[mat == 0] = 1
    # mat[mat == 2] = 0
    # builder = config_builder(mat)
    # builder.add_pullblocks()
    # builder.view()
    # builder.save_lammps("sheet", ext = f"ruptest", path = '../friction_simulation')
    
    
  
  
    #######################################################
    

    