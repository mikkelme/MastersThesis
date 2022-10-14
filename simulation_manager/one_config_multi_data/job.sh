#!/bin/bash

#SBATCH --job-name=great4
#
#SBATCH --partition=normal
#
#SBATCH --ntasks=16
#
#SBATCH --nodes=1
#

# Sbatch version here
mpirun -n 16 lmp -in produce_reset_files.in

# Then add
wait # wait till restart files is generated

# Then execute all restarts files as fast as possible

## ONE possibility 
# sbatch --array=1-100 MyScript
# will run 100 copies of MyScript, 
# setting the environment variable 
# $SLURM_ARRAY_TASK_ID to 1, 2, ..., 100 in turn.


## Else do something like 
run 1 F_N = 1
run 1 F_N = 1
run 1 F_N = 1
run 2 F_N = 2
run 2 F_N = 2
run 2 F_N = 2
run 3 F_N = 3
run 3 F_N = 3
run 3 F_N = 3