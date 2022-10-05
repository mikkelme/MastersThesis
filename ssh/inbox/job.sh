#!/bin/bash
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=debugGPU
#SBATCH --cpus-per-task=2

echo $CUDA_VISIBLE_DEVICES
mpirun -n 1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in ./cluster_test.in

# mpirun -n 1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in ./reproduce_GPU_bug.in

# with just "-sf gpu" added, but using "-pk gpu 0 neigh no" is a
# neigh full (try this also)

# mpirun -n 1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half binsize 7.5 -in ./run.in

# Replace 1 everywhere with the number of GPUs.
# binsize 7.5 is for SiO2 Vashishta, remove or adjust otherwise.


