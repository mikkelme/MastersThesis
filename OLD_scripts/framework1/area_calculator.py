""" Quick python code to calculate area 
    using distance between sheet and lower block 
    with positions read from seperate dump files. 
    
    If succesfull I think this shoud be implemented as a C++ script.  """

import numpy as np
import matplotlib.pyplot as plt


def main(sheet_dump, lower_block_dump):
    contact_threshold = 2 # [Å]
    contact_threshold = 3
    sheet_infile = open(sheet_dump, "r")
    lb_infile = open(lower_block_dump, "r")

    timestep = []
    min_distances = []
    # contact_pct = [] # pct 

    while True: # Timestep loop
        # --- Sheet positions --- #

        info = [sheet_infile.readline() for i in range(9)]
        if info[0] == '': break
        sheet_timestep = int(info[1].strip("\n"))
        sheet_num_atoms = int(info[3].strip("\n"))
        sheet_atom_pos = np.zeros((sheet_num_atoms, 3))
        print(f"\rtimestep = {sheet_timestep}", end = "")


        for i in range(sheet_num_atoms): # sheet atom loop
            line = sheet_infile.readline() # id type x y z vx vy vz
            words = np.array(line.split(), dtype = float)
            sheet_atom_pos[i] = words[2:5]

        # --- Lower block positions --- #
        info = [lb_infile.readline() for i in range(9)]
        lb_timestep = int(info[1].strip("\n"))
        lb_num_atoms = int(info[3].strip("\n"))
        lb_atom_pos = np.zeros((lb_num_atoms, 3))

        for i in range(lb_num_atoms): # lb atom loop
            line = lb_infile.readline() # id type x y z vx vy vz
            words = np.array(line.split(), dtype = float)
            lb_atom_pos[i] = words[2:5]

        # --- Calculate distance
        # Verify mathcing timesteps
        timestep_error = f"sheet timestep = {sheet_timestep} != lower block timestep = {lb_timestep}"
        assert sheet_timestep == lb_timestep, timestep_error

        # Minimum distance
        min_distance = np.min(np.linalg.norm(sheet_atom_pos - lb_atom_pos[:,None], axis=-1), axis = 0)
        
        # Append results

        timestep.append(sheet_timestep)
        min_distances.append(min_distance)
        # contact_pct.append(np.count_nonzero(min_distance < contact_threshold)/lb_num_atoms)

        # if timestep[-1] >= 8000: break

    print()
    timestep = np.array(timestep)
    min_distances = np.array(min_distances)

    threshold = [5, 4, 3, 2, 1]


    threshold = [4, 3, 2, 1.5, 1.25]
    for t in threshold:
        contact_pct = np.count_nonzero(min_distances < t, axis = 1)/sheet_num_atoms
        plt.plot(timestep, contact_pct, "-o", markersize = 3, label = f"threshold = {t} Å")

    plt.vlines(8000, 0, 0.2, linestyle = "--", color = "k", label = "Stretch begin")
    plt.legend()
    plt.xlabel("timestep")
    plt.ylabel("contact count (%)")
    plt.show()

    # print(distances)

if __name__ == "__main__":
    sheet_dump = "data/stretch_mul3x3_dis5_sheet.data"
    lower_block_dump = "data/stretch_mul3x3_dis5_lower_block.data"

    main(sheet_dump, lower_block_dump)
