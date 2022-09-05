""" Quick python code to calculate area 
    using distance between sheet and lower block 
    with positions read from seperate dump files. 
    
    If succesfull I think this shoud be implemented as a C++ script.  """

import numpy as np


def main(sheet_dump, lower_block_dump):
    contact_threshold = 2 # [Å]
    sheet_infile = open(sheet_dump, "r")
    lb_infile = open(lower_block_dump, "r")

    timestep = []
    num_contact = [] # pct 

    while True: # Timestep loop
        # Sheet positions        
        info = [sheet_infile.readline() for i in range(9)]
        sheet_timestep = int(info[1].strip("\n"))
        sheet_num_atoms = int(info[3].strip("\n"))
        sheet_atom_pos = np.zeros((sheet_num_atoms, 3))

        for i in range(sheet_num_atoms): # sheet atom loop
            line = sheet_infile.readline() # id type x y z vx vy vz
            words = np.array(line.split(), dtype = float)
            sheet_atom_pos[i] = words[2:5]

        # lb positions
        info = [lb_infile.readline() for i in range(9)]
        lb_timestep = int(info[1].strip("\n"))
        lb_num_atoms = int(info[3].strip("\n"))
        lb_atom_pos = np.zeros((lb_num_atoms, 3))

        for i in range(lb_num_atoms): # lb atom loop
            line = lb_infile.readline() # id type x y z vx vy vz
            words = np.array(line.split(), dtype = float)
            lb_atom_pos[i] = words[2:5]


        timestep_error = f"sheet timestep = {sheet_timestep} != lower block timestep = {lb_timestep}"
        assert sheet_timestep == lb_timestep, timestep_error
        timestep.append(sheet_timestep)

        # Calculate minimum positions
        min_distance = np.min(np.linalg.norm(sheet_atom_pos - lb_atom_pos[:,None], axis=-1), axis = 0)
        num_contact.append(np.count_nonzero(min_distance < contact_threshold))


        
        print(timestep[-1], num_contact[-1])





    # print(distances)

if __name__ == "__main__":
    sheet_dump = "sheet.data"
    lower_block_dump = "lower_block.data"

    main(sheet_dump, lower_block_dump)
