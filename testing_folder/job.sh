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

mpirun -n 1 lmp -pk kokkos newton on neigh half -k on g 1 -sf kk -in reproduce.in
