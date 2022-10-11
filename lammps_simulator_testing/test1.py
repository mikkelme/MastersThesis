# from lammps_simulator import sim

# sim.set_input_script("script.in")
# sim.run(num_procs=4, lmp_exec="lmp_mpi")

import sys
sys.path.append('../../lammps-simulator') # parent folder: MastersThesis
# from ../lammps_simulator import Simulator
from lammps_simulator import *
import subprocess

command = ['ssh', 'egil', 'mkdir', 'sim_ssh_test']

# sim = Simulator(directory = "sim_ssh_test", overwrite=True)
sim = Simulator(directory = "sim_ssh_test", overwrite=True, ssh = 'egil')
# sim.copy_to_wd("../potentials/si.sw", "../potentials/CH.airebo")
sim.set_input_script("script.in")
slurm_args = {'job_name':'cpu', 'partition':'normal', 'ntasks':16, 'nodes':1}
sim.run(num_procs=4, lmp_exec="lmp", slurm=True, slurm_args=slurm_args)
# sim.run(num_procs=4, lmp_exec="lmp_mpi")


# subprocess.Popen(['ssh', 'egil', 'mkdir', 'sim_ssh_test'])
# output = subprocess.run("ssh egil mkdir sim_ssh_test", shell=True)






# # kopiere filer
# subprocess.Popen(['rsync', '-av', 'filename', 'egil:~/wd'])

# # opprette mapper
# subprocess.Popen(['ssh', 'egil', '"mkdir', 'dir"'])

# # kjøre script
# job_id = subprocess.check_output(['ssh', 'egil', '"cd', wd, '&&', 'sbatch', f'{jobscript}"'])


# sim.set_input_script("../script.in", copy=False) # set relative path instead of copying
# sim.create_subdir("subdir1", "subdir2") # creating sub directories

# sim.copy_to_wd("../potentials/si.sw", "../potentials/CH.airebo")
# sim.set_input_script("script.in")










#####################

# slurm_args = {'job_name':'cpu', 'partition':'normal', 'ntasks':16, 'nodes':1}
# sim.run(num_procs=4, lmp_exec="lmp")#, slurm=True, slurm_args=slurm_args)
# print(sim.var)


# from lammps_simulator import Simulator

# sim = Simulator(directory = "egil:./simulation_test")
# sim.copy_to_wd("../potentials/si.sw", "../potentials/CH.airebo")

# lmp_vars = {'var1': 1.0, 'var2': "string?", }
# sim.set_input_script("script.in", **lmp_vars)

# slurm_args = {'job_name':'cpu', 'partition':'normal', 'ntasks':16, 'nodes':1}
# sim.run(num_procs=4, lmp_exec="lmp", slurm=True, slurm_args=slurm_args)