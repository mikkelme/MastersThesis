from analysis_utils import *


def detect_rupture(filename):
    timestep, hist = read_histogram(filename)
    data_freq = timestep[1] - timestep[0]
    cnum = hist[:, :, 1]
    rupture_flags = np.full(hist.shape[1], np.nan)
    
    # Filter
    target_window = 5000
    for i in range(hist.shape[1]):
        cnum[:,i] = savgol_filter(int(target_window/data_freq), 5, cnum[:,i])[0]
    
    deltacnum = np.full(np.shape(cnum), np.nan)
    deltacnum[1:-1] = cnum[2:] - cnum[:-2]
    
    
    cut = 10
    thresshold_ratio = 0.5
    for i in range(2, hist.shape[1]):
        minpeak = timestep[cut + np.argmin(deltacnum[cut:-cut, i])], np.min(deltacnum[cut:-cut, i])
        maxpeak = timestep[cut + np.argmax(deltacnum[cut:-cut, i])], np.max(deltacnum[cut:-cut, i])
        
        # Check if significant cnum negative dip accurs after positive peak
        flags = (maxpeak[0] < minpeak[0], abs(minpeak[1]) > abs(maxpeak[1]) * thresshold_ratio)
        rupture_flags[i] = flags[0] & flags[1]
    
    rupture_score = np.mean(rupture_flags[~np.isnan(rupture_flags)])
    return rupture_score
    
    # plt.figure(num = 0)
    # for i in range(hist.shape[1]):
    #     plt.plot(timestep[cut:-cut], cnum[cut:-cut, i], label = f'center = {hist[0,i,0]}')
    # plt.vlines(31000, np.min(cnum[1:-1, :]), np.max(cnum[1:-1, :]), color = 'k', linestyle = "--")
    # plt.legend()
    
    
    # plt.figure(num = 1)
    # for i in range(hist.shape[1]):
    #     plt.plot(timestep[cut:-cut], deltacnum[cut:-cut, i], label = f'center = {hist[0,i,0]}')
    # plt.vlines(31000, np.min(deltacnum[1:-1, :]), np.max(deltacnum[1:-1, :]), color = 'k', linestyle = "--")
    # plt.legend()
    
    # plt.show()




if __name__ == "__main__":
    
    filename = "../Data/chist_samples/rupture/cut_25stretch_chist_extreme.txt"
    filename = "../Data/chist_samples/rupture/cut_25stretch_chist_old.txt"
    filename = "../Data/chist_samples/rupture/cut_30stretch_chist.txt"
    
    filename = "../Data/chist_samples/intact/cut_25stretch_chist_FN300.txt"
    filename = "../Data/chist_samples/intact/cut_25stretch_chist.txt"
    rupture_score = detect_rupture(filename)
    print(f"rupture_score = {rupture_score}")
    