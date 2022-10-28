from analysis_utils import *


def detect_rupture_old(filename, stretchfile = None, check = False):
    # --- Settings --- #
    # Filter
    target_window_length = 2000 # [Timesteps]
    polyorder = 5
    
    # Detection 
    cut = 10 # cut-off from data ends due to savgol filter artifacts [indexes]
    thresshold_ratio = 0.7 # |minpeak| > ratio * |maxpeak| => RUPTURE
    std_tol = 1000 # Tolerance for std between minpeak location in timesteps
    
    # --- Get data --- #
    timestep, hist = read_histogram(filename)
    data_freq = timestep[1] - timestep[0]
    cnum = hist[:, :, 1]
    rupture_flags = np.full(hist.shape[1], np.nan)
    stdpeak = np.full(hist.shape[1], np.nan)
    minmax_peakidx = np.full((hist.shape[1], 2), np.nan)
    
    if stretchfile != None:
        timestep_merge, hist_merge = read_histogram(stretchfile)
        data_freq_merge = timestep_merge[1] - timestep_merge[0]
        cnum_merge = hist_merge[:, :, 1]
        
        assert(data_freq == data_freq_merge)
        assert(cnum_merge.shape[1] == cnum.shape[1])
        cnum = np.concatenate((cnum_merge, cnum))
        timestep = np.concatenate((timestep_merge, timestep))
    
    # Filter
    for i in range(hist.shape[1]):
        cnum[:,i] = savgol_filter(int(target_window_length/data_freq), polyorder, cnum[:,i])[0]
        
    deltacnum = np.full(np.shape(cnum), np.nan)
    deltacnum[1:-1] = cnum[2:] - cnum[:-2]
    
    
    # --- Detection --- #
    # minmmax peaks
    for i in range(2, hist.shape[1]): # (Nothing implemeted for i = 0, 1)
        minmax_peakidx[i, 0], minpeak = timestep[cut + np.argmin(deltacnum[cut:-cut-1, i])], np.min(deltacnum[cut:-cut-1, i])
        minmax_peakidx[i, 1], maxpeak = timestep[cut + np.argmax(deltacnum[cut:-cut-1, i])], np.max(deltacnum[cut:-cut-1, i])
            
        # Check if significant cnum minpeak accurs after maxpeak
        flags = (minmax_peakidx[i,1] < minmax_peakidx[i,0], abs(minpeak) > abs(maxpeak) * thresshold_ratio)
        rupture_flags[i] = flags[0] & flags[1]
        stdpeak[i] = np.std(deltacnum[cut:-cut-1, i])
            
    A = maxpeak/stdpeak
    B = minpeak/stdpeak
   
    A = A[~np.isnan(A)]
    B = B[~np.isnan(B)]
    ratio = B/A
    print(A)
    print(B)
    print(ratio)        
        
        
    
    # Minpeak idx std check (counteract confusion with no stretch simulations)
    std = minmax_peakidx[~np.isnan(minmax_peakidx)].reshape(-1, 2).std(0)
    if std[0] > std_tol: # Rupture flags <-- 0
        rupture_flags[rupture_flags == 1] = 0
    
    # Final average score 
    rupture_score = np.mean(rupture_flags[~np.isnan(rupture_flags)])
    
    
    if check: # Show plots and flags
        print(f'Rupture flags: {rupture_flags}')
        plt.figure(num = 0)
        for i in range(hist.shape[1]):
            plt.title("coordination number")
            plt.plot(timestep[cut:-cut-1], cnum[cut:-cut-1, i], label = f'center = {hist[0,i,0]}')
            plt.xlabel("Timestep")
            plt.ylabel("$cnum")
        plt.legend()
        
        plt.figure(num = 1)
        for i in range(hist.shape[1]):
            plt.title("$\Delta$ coordination number")
            plt.plot(timestep[cut:-cut-1], deltacnum[cut:-cut-1, i], label = f'center = {hist[0,i,0]}')
            plt.xlabel("Timestep")
            plt.ylabel("$\Delta$ cnum")
        plt.legend()
        
    return rupture_score

def detect_rupture(filename, stretchfile = None, check = False):
    # --- Settings --- #
    # Filter
    target_window_length = 2000 # [Timesteps]
    polyorder = 5
    
    # Detection 
    cut = 0 # cut-off from data ends due to savgol filter artifacts [indexes]
    min_align_tol = 1000 # Tolerance for std between minpeak location in timesteps
    max_align_tol = 10000 # Tolerance for std between minpeak location in timesteps
    ratio_tol = 0.7 # in the case of maxpeak alignment
    minpeak_tol = -3 # in the case of no max peak alignment 
    
    # --- Get data --- #
    timestep, hist = read_histogram(filename)
    data_freq = timestep[1] - timestep[0]
    cnum = hist[:, :, 1]
    rupture_flags = np.full(hist.shape[1], np.nan)
    
    if stretchfile != None:
        timestep_merge, hist_merge = read_histogram(stretchfile)
        data_freq_merge = timestep_merge[1] - timestep_merge[0]
        cnum_merge = hist_merge[:, :, 1]
        
        seam = timestep[0]
        assert(data_freq == data_freq_merge)
        assert(cnum_merge.shape[1] == cnum.shape[1])
        
        cnum = np.concatenate((cnum_merge[timestep_merge < seam], cnum))
        timestep = np.concatenate((timestep_merge[timestep_merge < seam], timestep))
       
        
    # Filter
    for i in range(hist.shape[1]):
        cnum[:,i] = signal.savgol_filter(cnum[:,i], int(target_window_length/data_freq), polyorder)
        
        
    deltacnum = np.full(np.shape(cnum), np.nan)
    deltacnum[1:-1] = cnum[2:] - cnum[:-2]
    
    
    # --- Detection --- #
    minpeak_step = np.full((hist.shape[1]), np.nan)
    minpeak = np.full((hist.shape[1]), np.nan)
    maxpeak_step = np.full((hist.shape[1]), np.nan)
    maxpeak = np.full((hist.shape[1]), np.nan)
    stdpeak = np.full((hist.shape[1]), np.nan)
    
    
    for i in range(2, hist.shape[1]): # (Nothing implemeted for i = 0, 1)    
        
        target = ~np.isnan(deltacnum[cut:-cut-1, i])
        array = deltacnum[cut:-cut-1, i][~np.isnan(deltacnum[cut:-cut-1, i])]
        
        
        minpeak_step[i], minpeak[i] = timestep[cut + np.argmin(array)], np.min(array)
        maxpeak_step[i], maxpeak[i] = timestep[cut + np.argmax(array)], np.max(array)
        stdpeak[i] = np.std(array)
        
        # maxpeak_step[i], maxpeak[i] = timestep[cut + np.argmax(deltacnum[cut:-cut-1, i])], np.max(deltacnum[cut:-cut-1, i], initial = 0, where=~np.isnan(deltacnum[cut:-cut-1, i]))
        # stdpeak[i] = np.std(deltacnum[cut:-cut-1, i], where=~np.isnan(deltacnum[cut:-cut-1, i]))
        
        
        # minpeak_step[i], minpeak[i] = timestep[cut + np.argmin(deltacnum[cut:-cut-1, i]), initial = 0, where=~np.isnan(deltacnum[cut:-cut-1, i])], np.min(deltacnum[cut:-cut-1, i], initial = 0, where=~np.isnan(deltacnum[cut:-cut-1, i]))
        # maxpeak_step[i], maxpeak[i] = timestep[cut + np.argmax(deltacnum[cut:-cut-1, i])], np.max(deltacnum[cut:-cut-1, i], initial = 0, where=~np.isnan(deltacnum[cut:-cut-1, i]))
        # stdpeak[i] = np.std(deltacnum[cut:-cut-1, i], where=~np.isnan(deltacnum[cut:-cut-1, i]))
    
    
    minpeak_ratio = minpeak[stdpeak > 0]/stdpeak[stdpeak > 0]
    maxpeak_ratio = (maxpeak[stdpeak > 0]/stdpeak[stdpeak > 0])
    
    minpeak_alignment = np.std(minpeak_step[~np.isnan(minpeak_step)]) < min_align_tol # Stretch present
    maxpeak_alignment = np.std(maxpeak_step[~np.isnan(maxpeak_step)]) < max_align_tol # Possible rip present
        
    
    if maxpeak_alignment:
        # check if minpeak magnitude significant
        magnitude = minpeak_ratio/maxpeak_ratio < ratio_tol
    
    else:
        # This is probably never applied due to min_align check afterwards
        magnitude = minpeak_ratio < minpeak_tol
        
        
    rupture_flags = minpeak_alignment & magnitude
    rupture_score = np.mean(rupture_flags[~np.isnan(rupture_flags)])
    
    
    #### Can also check that min comes after max
    
    if check: # Show plots and flags
        print(f'Alignment (min, max) = ({minpeak_alignment}, {maxpeak_alignment})')
        print(f'Magnitude = {magnitude}')
        print(f'Rupture flags: {rupture_flags}')
        plt.figure(num = 0)
        for i in range(hist.shape[1]):
            plt.title("coordination number")
            plt.plot(timestep[cut:-cut-1], cnum[cut:-cut-1, i], label = f'center = {hist[0,i,0]}')
            plt.xlabel("Timestep")
            plt.ylabel("cnum")
        plt.legend()
        
        plt.figure(num = 1)
        for i in range(hist.shape[1]):
            plt.title("$\Delta$ coordination number")
            plt.plot(timestep[cut:-cut-1], deltacnum[cut:-cut-1, i], label = f'center = {hist[0,i,0]}')
            plt.xlabel("Timestep")
            plt.ylabel("$\Delta$ cnum")
        plt.legend()
        plt.show()
    
    
    return rupture_score




if __name__ == "__main__":
    
    filenames = ["../Data/NG4_newpot_long/cut_nostretch/_cut_nostretch_chist.txt"]
    filenames = ["../Data/NG4_newpot_long/cut_20stretch/_cut_20stretch_chist.txt"]
    filenames = get_files_in_folder('../Data/NG4_newpot_long/', ext = "chist.txt")
    
    
    filenames = get_files_in_folder('../Data/OCMD_newpot/stretch.10992_folder', ext = "chist.txt")
    filenames += get_files_in_folder('../Data/OCMD_newpot/stretch.10564_folder', ext = "chist.txt")
    filenames += get_files_in_folder('../Data/OCMD_newpot/stretch.5428_folder', ext = "chist.txt")
    
    # filenames = ['../Data/OCMD_newpot/stretch.8424_folder/job4/_tmp_chist.txt']# ../Data/OCMD_newpot/stretch__default_chist.txt

    # filenames = get_files_in_folder('../Data/chist_samples/intact', ext = "chist.txt")
    
    
    stretchfile = '../Data/OCMD_newpot/stretch__default_chist.txt'
    # stretchfile = None
    for filename in filenames:
        rupture_score = detect_rupture(filename, stretchfile = stretchfile, check = False)
        print(f"filename: {filename}, rupture_score = {rupture_score}")
        print()
        plt.show()


### Work on rupture detection still....
### If stretch files is used it most often predicts rupture....