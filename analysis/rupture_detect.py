from analysis_utils import *


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

    # Append stretchfile in front
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
        
    # coordination number change
    deltacnum = np.full(np.shape(cnum), np.nan)
    deltacnum[1:-1] = (cnum[2:] - cnum[:-2]) / (2*data_freq)
    
    
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
        
      
    
    minpeak_ratio = minpeak[stdpeak > 0]/stdpeak[stdpeak > 0]
    maxpeak_ratio = (maxpeak[stdpeak > 0]/stdpeak[stdpeak > 0])
    
    notnan = ~np.isnan(minpeak_step) & ~np.isnan(maxpeak_step)
    minpeak_alignment = np.std(minpeak_step[notnan]) < min_align_tol # Stretch present
    maxpeak_alignment = np.std(maxpeak_step[notnan]) < max_align_tol # Possible rip present
    peakorder = np.mean(minpeak_step[notnan]) > np.mean(maxpeak_step[notnan]) 
        
    
    
    if maxpeak_alignment:
        # check if minpeak magnitude significant
        magnitude = abs(minpeak_ratio/maxpeak_ratio) > ratio_tol
    
    else:
        # This is probably not used so much due to min_align check afterwards
        magnitude = minpeak_ratio < minpeak_tol
        
        
    rupture_flags = peakorder & minpeak_alignment & magnitude
    rupture_score = np.mean(rupture_flags[~np.isnan(rupture_flags)])
    
    
    
    if check: # Show plots and flags
        print(f'Alignment (min, max) = ({minpeak_alignment}, {maxpeak_alignment})')
        print(f'Peakorder: min = {np.mean(minpeak_step[notnan])} > max = {np.mean(maxpeak_step[notnan])} => {peakorder} ')
        print(f'Magnitude = {magnitude}')
        print(f'Rupture flags: {rupture_flags}')
        print(f'abs(minpeak_ratio/maxpeak_ratio): {abs(minpeak_ratio/maxpeak_ratio)}')
        plt.figure(num = 0)
        for i in range(hist.shape[1]):
            plt.title("coordination number")
            plt.plot(timestep[cut:-cut-1], cnum[cut:-cut-1, i], color = color_cycle(i), label = f'center = {hist[0,i,0]}')
            if stretchfile != None:
                plt.plot(timestep_merge, cnum_merge[:, i], color = color_cycle(i))
                if i == 0:
                    plt.vlines(seam, np.min(cnum_merge), np.max(cnum_merge), linestyle = '--', color = 'k')
            plt.xlabel("Timestep")
            plt.ylabel("cnum")
        plt.legend()
        
        plt.figure(num = 1)
        for i in range(hist.shape[1]):
            plt.title("$\Delta$ coordination number")
            plt.plot(timestep[cut:-cut-1], deltacnum[cut:-cut-1, i], color = color_cycle(i), label = f'center = {hist[0,i,0]}')
            plt.xlabel("Timestep")
            plt.ylabel("$\Delta$ cnum")
        plt.legend()
        plt.show()
    
    
    return rupture_score




if __name__ == "__main__":
    filenames = []
    # filenames = ["../Data/NG4_newpot_long/cut_nostretch/_cut_nostretch_chist.txt"]
    # filenames = ["../Data/NG4_newpot_long/cut_20stretch/_cut_20stretch_chist.txt"]
    # filenames = get_files_in_folder('../Data/NG4_newpot_long/', ext = "chist.txt")
    

    # filenames = get_files_in_folder('../Data/OCMD_newpot/stretch.10992_folder', ext = "chist.txt")
    # filenames += get_files_in_folder('../Data/OCMD_newpot/stretch.10564_folder', ext = "chist.txt")
    # filenames += get_files_in_folder('../Data/OCMD_newpot/stretch.5428_folder', ext = "chist.txt")
    

    # filenames = get_files_in_folder('../Data/chist_samples/intact', ext = "chist.txt")
    
   
    
    # filenames = get_files_in_folder('../Data/OCMD_', ext = "ext_chist.txt")
    # filenames = ['../Data/OCMD_newpot/stretch.10992_folder/job0/_tmp_chist.txt'] 
    # filenames = ['../Data/OCMD_newpot/stretch.5856_folder/job0/_tmp_chist.txt'] # This looked like a rupture, but download the dump file and check <--------
    
    
    # stretchfile = '../Data/OCMD_newpot/stretch__default_chist.txt'
    # filenames = ['../Data/OCMD_newpot/stretch__default_chist.txt'] 
    
    
    # filenames = ['../Data/OCMD_newpot/stretch.6712_folder/job0/_tmp_chist.txt'] 
    # stretchfile = '../Data/OCMD_newpot/stretch__default_chist.txt'
    
    
    
    
    filenames += get_files_in_folder('../Data/NG4_GPU/', ext = "chist.txt")
    stretchfile = None
    
    ###########################################
    
    # filenames = get_files_in_folder('../Data/multi_short', ext = "ext_chist.txt")
    # filenames = ['../Data/multi_short/stretch_6712_folder/job0/sheet_ext_chist.txt']
    # stretchfile = '../Data/multi_short/sheet__default_chist.txt'
    
    ###########################################
    
    
    # filenames = get_files_in_folder('../Data/rupture_test', ext = "ext_chist.txt")
    # filenames = ['../Data/rupture_test/stretch_6998_folder/job0/sheet_ext_chist.txt']
    # stretchfile = '../Data/rupture_test/sheet__default_chist.txt'
    
    # filenames = ['../Data/rupture_test/sheet_local_chist.txt']
    # stretchfile = None
    
    
    
    for filename in filenames:
        rupture_score = detect_rupture(filename, stretchfile = stretchfile, check = False)
        print(f"filename: {filename}, rupture_score = {rupture_score}")
        print()
        # plt.show()
        
        
        
