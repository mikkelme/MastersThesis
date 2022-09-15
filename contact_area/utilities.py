import numpy as np

def read_stretch_file(filename):
    print("# --- Reading stretch file --- # ")
    print(f"Filename = \"{filename}\"")
    timestep, stretch_pct, ylow_force, yhigh_force = np.loadtxt(filename, delimiter = " ", unpack = True)
    print()
    return timestep, stretch_pct, ylow_force, yhigh_force 



def get_stretch_timestamps(stretch_file):
    timestep, stretch_pct, ylow_force, yhigh_force = read_stretch_file(stretch_file)
    delta_stretch = stretch_pct[1:] - stretch_pct[:-1] 
    diff = np.zeros((2, len(timestep)-2))
    diff[0] = np.abs(stretch_pct[1:-1] - stretch_pct[:-2])    # backwards
    diff[1] = np.abs(stretch_pct[2:] - stretch_pct[1:-1]) # forward


    timestamps  = np.argwhere(np.logical_and(np.min(diff, axis = 0) == 0, np.max(diff, axis = 0) != 0)).ravel() 
    step_correction = diff[:, timestamps][0, :] == 0 # + 1 if starting stretch
    timestamps += step_correction + 1 # +1 for diff starting from timestep idx 1

    # for i in range(len(timestep)-2):
    #     print(timestep[i+1], diff[0, i], diff[1, i ], np.min(diff[:,i], axis = 0) == 0 and np.max(diff[:,i], axis = 0) != 0)

    if len(timestamps) == 0:
        return timestamps
    else:
        return timestep[timestamps]