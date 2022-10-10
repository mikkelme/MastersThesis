# from lammps_simulator import sim

# sim.set_input_script("script.in")
# sim.run(num_procs=4, lmp_exec="lmp_mpi")


from lammps_simulator import Simulator

sim = Simulator(directory = "simulation", overwrite=True)
# sim.set_input_script("../script.in", copy=False) # set relative path instead of copying
# sim.create_subdir("subdir1", "subdir2") # creating sub directories

sim.copy_to_wd("../potentials/si.sw", "../potentials/CH.airebo")


lmp_vars = {'var1': 1.0, 'var2': "tekst", 'var_new': 42}
sim.set_input_script("script.in", **lmp_vars)

# slurm_args = {'job_name':'cpu', 'partition':'normal', 'ntasks':16, 'nodes':1}
sim.run(num_procs=4, lmp_exec="lmp", slurm=True, slurm_args=slurm_args)
# print(sim.var)




# from lammps_simulator import Simulator

# sim = Simulator(directory = "egil:./simulation_test")
# sim.copy_to_wd("../potentials/si.sw", "../potentials/CH.airebo")

# lmp_vars = {'var1': 1.0, 'var2': "string?", }
# sim.set_input_script("script.in", **lmp_vars)

# slurm_args = {'job_name':'cpu', 'partition':'normal', 'ntasks':16, 'nodes':1}
# sim.run(num_procs=4, lmp_exec="lmp", slurm=True, slurm_args=slurm_args)