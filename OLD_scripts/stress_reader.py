from ovito.io import import_file #, export_file
import matplotlib 
matplotlib.use('Agg') # Can only do offscreen plotting for somw reason
# matplotlib.use('qtAgg')
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np

# from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar # original bug detection


def read_stress(filename):
    pipeline = import_file(filename)
    num_frames = pipeline.source.num_frames
    frames = np.linspace(0, num_frames-1, num_frames).astype('int')
    min_stress = []; max_stress = []
    for frame in frames:
        data = pipeline.compute(frame = frame)
        # pos = data.particles.positions[...]
        stress = data.particles['v_stress_atom'][...]
        
        min_stress.append([min(stress)])
        max_stress.append([max(stress)])
    
    return np.array(frames), np.array(min_stress), np.array(max_stress)
   
    # print(data)
    
    # fp = pipeline.num_frames()
    # stress = data.particles.v_stress_atom[...]
    # color_property = data.particles['Particle Type']
    # print(color_property)
    # print(data.particles_)
    


if __name__ == "__main__":
    filename = 'stress_dump.data'
    
    infile = open('max_stress.txt', 'r')
    infile.readline()
    infile.readline()
    step = []
    max_stress_file = []
    for line in infile:
        words = infile.readline().split(' ')
        try:
            step.append(float(words[0]))
            max_stress_file.append(float(words[1]))
        except:
            break
        
    step = np.array(step)
    max_stress_file = np.array(max_stress_file)
    ### Use this as input ^^^
        
    
    
    frames, min_stress, max_stress = read_stress(filename)
    plt.plot(step/100, max_stress_file)
    plt.plot(frames, min_stress, "o") 
    plt.plot(frames, max_stress, "o") 
    plt.vlines(45, min_stress.min(), max_stress_file.max(), linestyle = '--', color = 'k')
    plt.savefig('test.pdf')
    # plt.show()
