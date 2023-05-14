import os
import numpy as np

folder = 'topview'
# folder = 'sideview'
prefix = 'flip'
ext = '.png'
start_num = 8


numbers = []
# Get info
for file in os.listdir(folder):
    if file[:len(prefix)] != prefix:
        continue
    words = file.split(prefix)
    num = int(words[1].lstrip('0').rstrip(ext))
    numbers.append(num)
    print(file)

    
numbers = np.sort(numbers)

# Rename and duplicate for reverse 
for file in os.listdir(folder):
    if file[:len(prefix)] != prefix:
        continue
    words = file.split(prefix)
    num = int(words[1].lstrip('0').rstrip(ext))
    
    # New numbers
    new_num = np.argmin(abs(num - numbers)) + start_num
    sec_num = 2*(len(numbers) + start_num) - new_num -1
    
    # Commands
    os.system(f'cp {os.path.join(folder, file)} {os.path.join(folder, prefix + str(sec_num) + ext)}')
    os.system(f'mv {os.path.join(folder, file)} {os.path.join(folder, prefix + str(new_num) + ext)}')
    
    
