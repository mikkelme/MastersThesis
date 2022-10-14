#!/bin/bash

#SBATCH --job-name=great4
#
#SBATCH --partition=normal
#
#SBATCH --ntasks=16
#
#SBATCH --nodes=1
#

mpirun -n 1 lmp -in produce_reset_files.in
