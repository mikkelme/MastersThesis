import os
import numpy as np

# Settings
type = 'topview'; start_num = 7
# type = 'sideview'; start_num = 8
freq = 4
unqiue_center = False
fill_around_center = 3


from_folder = f'{type}_full'
to_folder = f'{type}'
prefix = 'flip'
ext = '.png'
numbers = []

# Get numbers
for file in os.listdir(from_folder):
    if file[:len(prefix)] != prefix:
        continue
    words = file.split(prefix)
    num = int(words[1].lstrip('0').rstrip(ext))
    numbers.append(num)
    # print(file)

numbers = np.sort(numbers)
target = numbers[np.arange(0, len(numbers), freq)]

# Fill around center
fill = []
for num in reversed(numbers):
    if not num in target:
        fill.append(num)
    if len(fill) == fill_around_center:
        break
    
target = np.sort(np.concatenate((target, np.array(fill))))

# Rename and duplicate for reverse 
count = 0
for file in np.sort(os.listdir(from_folder)):
    if file[:len(prefix)] != prefix:
        continue
    words = file.split(prefix)
    num = int(words[1].lstrip('0').rstrip(ext))
    if num in target:
        count += 1
        new_num = np.argmin(abs(num - target)) + start_num
        sec_num = 2*(len(target) + start_num) - new_num -1
        print(count, new_num, sec_num, num == target[-1])
    
        os.system(f'cp {os.path.join(from_folder, file)} {os.path.join(to_folder, prefix + str(new_num) + ext)}')
        if num == target[-1]:
            if unqiue_center:
                continue
        os.system(f'cp {os.path.join(from_folder, file)} {os.path.join(to_folder, prefix + str(sec_num) + ext)}')
            
    
    
    
    
