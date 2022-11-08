#!/bin/bash

#SBATCH --job-name=NG4_GPU
#
#SBATCH --partition=normal
#
#SBATCH --ntasks=1
#
#SBATCH --cpus-per-task=2
#
#SBATCH --gres=gpu:1
#
#SBATCH --output=slurm.out
#

mpirun -n 1 lmp -pk kokkos newton on neigh half -k on g 1 -sf kk -in friction_procedure.in -var dt 0.001 -var config_data sheet_substrate_nocuts -var relax_time 5 -var stretch_speed_pct 0.05 -var stretch_max_pct 0.2 -var pause_time1 5 -var F_N 6.24150907 -var pause_time2 5 -var drag_dir_x 0 -var drag_dir_y 1 -var drag_speed 0.01 -var drag_length 30 -var K 1.8724527210000002 -var root .. -var out_ext nocut_20stretch
