#!/bin/bash

#SBATCH --job-name=great4
#
#SBATCH --partition=normal
#
#SBATCH --ntasks=16
#
#SBATCH --nodes=1
#

mpirun -n 1 lmp_mpi -in produce_reset_files.in -var dt 0.001 -var config_data sheet_substrate -var relax_time 5 -var stretch_speed_pct 0.05 -var stretch_max_pct 0.05 -var pause_time1 5 -var F_N 99.86414512 -var pause_time2 5 -var drag_dir_x 0 -var drag_dir_y 1 -var drag_speed 0.05 -var drag_length 30 -var K 1.8724527210000002 -var root .. -var out_ext _default

wait
    
for file in *.restart; do    
    [ -f "$file" ] || break    
    lmp_serial -in start_from_restart_file.in -var restart_file $file    
done
