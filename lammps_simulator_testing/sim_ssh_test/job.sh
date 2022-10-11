#!/bin/bash

#SBATCH --job_name=cpu
#
#SBATCH --partition=normal
#
#SBATCH --ntasks=16
#
#SBATCH --nodes=1
#

mpirun -n 4 lmp_mpi -in script.in