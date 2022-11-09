from analysis_utils import *

def read_contact_pct(filename):
    timestep, sheet_bond_pct, full_sheet_bond_pct = np.loadtxt(filename, unpack=True)
    
    plt.plot(timestep, sheet_bond_pct, label = "sheet bonds")
    plt.plot(timestep, full_sheet_bond_pct, label = "full_sheet bonds")
    plt.legend()
    plt.show()
    # print(data)



if __name__ == "__main__":
    filename = '../friction_simulation/my_simulation_space/bond_pct.txt'
    # filename = '../Data/multi_fast/stretch_19994_folder/job0/bond_pct.txt'
    read_contact_pct(filename)