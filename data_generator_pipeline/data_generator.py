import numpy as np
import os


from ase.build import graphene_nanoribbon


class data_generator:
    def __init__(self, mat):
        shape_error = f"SHAPE ERROR: Got matrix of shape {np.shape(mat)}, y-axis must be multiple of 2 and both nonzero integer."
        assert mat.shape[0]%1 == 0 and mat.shape[1]%1 == 0 and mat.shape[1]%2 == 0, shape_error
        assert mat.shape[0] > 0 and mat.shape[1] > 0, shape_error
        
        self.mat = mat
        self.Cdis = 1.42 # carbon-carbon distance [Å]

        self.dir = " "
        self.substrate = " "

        self.shape = np.shape(mat)        
    
        self.set_sheet_size()
        
        
    def set_sheet_size(self):
        """ Set sheet size according to configuration matrix shape """
        xlen = self.shape[0]
        ylen = self.shape[1]//2
        a = 3*self.Cdis/np.sqrt(3)
        
        self.Lx = a/6*np.sqrt(3) + (xlen-1) * a/2*np.sqrt(3)
        self.Ly = a/2 + (ylen-1) * a
        
    
        
    def get_substrate_size(self, stretch_pct, margins = (10, 10)):
        """ Get substrate size considered sretch and x,y-margines """
        Lx = self.Lx + 2*margins[0]
        Ly = self.Ly * (1 + stretch_pct) + 2*margins[1]
        return Lx, Ly
        
        
      


    

def read_configurations(folder):
    configs = []
    # print(f"Reading configurations | dir: {folder}")
    for file in os.listdir(folder):
        try:
            mat = np.load(os.path.join(folder,file))
            configs.append(mat)
            # print("√ |", file)
            
        except ValueError:
            pass
            # print("X |", file)
    return configs












if __name__ == "__main__":
    configs = read_configurations("../graphene_sheet/test_data")
    conf = data_generator(configs[0])
    print(conf.Lx, conf.Ly)
    print(conf.get_substrate_size(0.20))
    
    
 
    
    
    # gen = data_generator()
    # gen.read_configurations("../graphene_sheet/test_data" )
    # gen.get_sheet_size(gen.configs[0])