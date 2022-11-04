from analysis_utils import *

def read_contact_pct(filename):
    timestep, sheet_bond_pct, full_sheet_bond_pct = np.loadtxt(filename, unpack=True)
    
    plt.plot(timestep, sheet_bond_pct, label = "sheet bonds")
    plt.plot(timestep, full_sheet_bond_pct, label = "full_sheet bonds")
    plt.legend()
    plt.show()
    # print(data)



if __name__ == "__main__":
    filename = '../friction_simulation/bond_pct.txt'
    read_contact_pct(filename)