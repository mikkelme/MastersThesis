### Scripts for building the silicon substrate 


import sys
sys.path.append('../') # parent folder: MastersThesis


from graphene_sheet.build_utils import *

from ase.lattice.cubic import FaceCenteredCubicFactory
class DiamondFactory(FaceCenteredCubicFactory):
    """A factory for creating diamond lattices."""
    xtal_name = 'diamond'
    bravais_basis = [[0, 0, 0], [0.25, 0.25, 0.25]]


def build_substrate(subpos, a_Si = 5.430953):
    """ Build substrate object in ASE with 
        minimum dimensions (Lx, Ly, Lz) = subpos  """
    
    # Get lattice size
    size = np.ceil([pos/a_Si for pos in subpos]).astype('int')
    
    # Build
    Diamond = DiamondFactory()
    atoms = Diamond(directions = [[1,0,0], [0,1,0], [0,0,1]], 
                    size = size, 
                    symbol = 'Si', 
                    pbc = (1,1,0), 
                    latticeconstant = a_Si)
    
    return atoms
    






if __name__ == "__main__":
    pass