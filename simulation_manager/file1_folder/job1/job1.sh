#!/bin/bash

#SBATCH --job-name=MULTI
#
#SBATCH --partition=normal
#
#SBATCH --ntasks=16
#
#SBATCH --nodes=1
#

mpirun -n 16 lmp -in ../../start_from_restart_file.in -var dt 0.001 -var config_data sheet_substrate -var relax_time 1 -var stretch_speed_pct 0.5 -var stretch_max_pct 0.05 -var pause_time1 5 -var F_N 124.8301814 -var pause_time2 5 -var drag_dir_x 0 -var drag_dir_y 1 -var drag_speed 0.05 -var drag_length 30 -var K 1.8724527210000002 -var root ../.. -var out_ext 1 -var restart_file ../file1.restart
