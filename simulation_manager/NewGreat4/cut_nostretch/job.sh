#!/bin/bash


mpirun -n 1 lmp_mpi -in run_friction_sim.in -var dt 0.001 -var config_data sheet_substrate -var relax_time 0.2 -var stretch_speed_pct 0.2 -var stretch_max_pct 0.0 -var pause_time1 0.2 -var F_N 6.24150907 -var pause_time2 0.2 -var drag_dir_x 0 -var drag_dir_y 1 -var drag_speed 0.05 -var drag_length 0.01 -var K 1.8724527210000002 -var root .. -var out_ext _cut_nostretch
