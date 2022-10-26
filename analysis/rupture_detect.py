from analysis_utils import *


def detect_rupture(filename):
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
    minmax_peakidx = np.full((hist.shape[1], 2), np.nan)
    
    # Filter
    for i in range(hist.shape[1]):
        cnum[:,i] = savgol_filter(int(target_window_length/data_freq), polyorder, cnum[:,i])[0]
        
    deltacnum = np.full(np.shape(cnum), np.nan)
    deltacnum[1:-1] = cnum[2:] - cnum[:-2]
    
    
    # --- Detection --- #
    for i in range(2, hist.shape[1]):
        # minpeak = timestep[cut + np.argmin(deltacnum[cut:-cut, i])], np.min(deltacnum[cut:-cut, i])
        # maxpeak = timestep[cut + np.argmax(deltacnum[cut:-cut, i])], np.max(deltacnum[cut:-cut, i])
        minmax_peakidx[i, 0], minpeak = timestep[cut + np.argmin(deltacnum[cut:-cut, i])], np.min(deltacnum[cut:-cut, i])
        minmax_peakidx[i, 1], maxpeak = timestep[cut + np.argmax(deltacnum[cut:-cut, i])], np.max(deltacnum[cut:-cut, i])
        
        
        # Check if significant cnum minpeak accurs after maxpeak
        # flags = (maxpeak[0] < minpeak[0], abs(minpeak[1]) > abs(maxpeak[1]) * thresshold_ratio)
        flags = (minmax_peakidx[i,1] < minmax_peakidx[i,0], abs(minpeak) > abs(maxpeak) * thresshold_ratio)
        rupture_flags[i] = flags[0] & flags[1]
        # minmax_peakidx[i] = minpeak[0], maxpeak[1]
    
    
    std = minmax_peakidx[~np.isnan(minmax_peakidx)].reshape(-1, 2).std(0)
    if std[0] > std_tol: # Noise flag
        rupture_flags[rupture_flags == 1] = 0
    
    rupture_score = np.mean(rupture_flags[~np.isnan(rupture_flags)])
    
    if False: # Verify manually
        print(rupture_flags)
        # print(np.std(deltacnum[cut:-cut, :], axis = 0))
        plt.figure(num = 0)
        for i in range(hist.shape[1]):
            plt.title("coordination number")
            plt.plot(timestep[cut:-cut], cnum[cut:-cut, i], label = f'center = {hist[0,i,0]}')
            plt.xlabel("Timestep")
            plt.ylabel("$cnum")
        # plt.vlines(31000, np.min(cnum[1:-1, :]), np.max(cnum[1:-1, :]), color = 'k', linestyle = "--")
        plt.legend()
        
        
        plt.figure(num = 1)
        for i in range(hist.shape[1]):
            plt.title("$\Delta$ coordination number")
            plt.plot(timestep[cut:-cut], deltacnum[cut:-cut, i], label = f'center = {hist[0,i,0]}')
            plt.xlabel("Timestep")
            plt.ylabel("$\Delta$ cnum")
        # plt.vlines(31000, np.min(deltacnum[1:-1, :]), np.max(deltacnum[1:-1, :]), color = 'k', linestyle = "--")
        plt.legend()
        
    return rupture_score




if __name__ == "__main__":
    
    filenames = [    "../Data/chist_samples/rupture/cut_25stretch_chist_extreme.txt", # Rupture
                    "../Data/chist_samples/rupture/cut_25stretch_chist_old.txt", # Rupture
                    "../Data/chist_samples/rupture/cut_30stretch_chist.txt", # Rupture
                    "../Data/chist_samples/intact/cut_25stretch_chist_FN300.txt", # No rupture
                    "../Data/chist_samples/intact/cut_25stretch_chist.txt"] # No rupture
    
    
    # filenames = ["../Data/NG4_newpot_long/cut_nostretch/_cut_nostretch_chist.txt"]
    # filenames = ["../Data/NG4_newpot_long/cut_20stretch/_cut_20stretch_chist.txt"]
    filenames = get_files_in_folder('../Data/NG4_newpot_long/', ext = "chist.txt")
    for filename in filenames:
        rupture_score = detect_rupture(filename)
        print(f"filename: {filename}, rupture_score = {rupture_score}")
        print()
        plt.show()
