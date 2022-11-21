from analysis_utils import *
fig_num = 0
def read_contact_pct(filename):
    plt.figure(num = fig_num)
    timestep, sheet_bond_pct, full_sheet_bond_pct = np.loadtxt(filename, unpack=True)
    
    plt.plot(timestep, sheet_bond_pct, label = "sheet bonds")
    plt.plot(timestep, full_sheet_bond_pct, label = "full_sheet bonds")
    plt.legend()



if __name__ == "__main__":
    # filename = '../Data/multi_fast/stretch_19994_folder/job0/bond_pct.txt'
    # filename = '../Data/contact_area_local/bond_pct.txt'
    
    
    
    filename = '../Data/BIG_MULTI_Ydrag/bond_pct.txt'
    read_contact_pct(filename); fig_num += 1
    filename = '../Data/BIG_MULTI_Ydrag/stretch_30974_folder/job0/bond_pct.txt'
    read_contact_pct(filename); fig_num += 1
    plt.show()