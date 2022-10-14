#!/bin/bash

#SBATCH --job-name=great4
#
#SBATCH --partition=normal
#
#SBATCH --ntasks=16
#
#SBATCH --nodes=1
#

# mpirun -n 1 lmp_mpi -in produce_reset_files.in
# wait

for file in *.restart; do
    [ -f "$file" ] || break
    lmp_serial -in start_from_restart_file.in -var restart_file $file
done