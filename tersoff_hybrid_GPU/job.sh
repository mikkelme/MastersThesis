#!/bin/bash

#SBATCH --job-name=debug_GPU
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

mpirun -n 1 lmp_KOKKOS -pk kokkos newton on neigh full -k on g 1 -sf kk -in friction_procedure.in
#mpirun -n 1 lmp_GPU -pk gpu 1 neigh no -sf gpu -in friction_procedure.in                                                       
