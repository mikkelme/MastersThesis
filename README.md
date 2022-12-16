# Master's thesis

## Logbook (work progress and problems occurring)
### 2022
### Week 34 (22/08 - 28/08)
- Started project 
- Decided on first subgoal: Investigating contact area as function of stretch (force)
- Working on indexing system for graphene sheet 
- Working on Lammps simulation pipeline

### Week 35 (29/08 - 04/09) - Working from Oslo
- Finsihed indexing system for each atom
- Created indexing system for positions between atoms (center elements)
- Created framework for deleting atoms when jumping between center elements 
- Working on code to produce what I call the [pop-up-pattern](https://www.seas.harvard.edu/news/2017/02/new-pop-strategy-inspired-cuts-not-folds), with the hope that it also buckles in 3D under strain on a molecular level.
- Added some functionality to the code for changing the sice of the pattern unit cell.
- Made first successfull simulation of the pop-up pattern being stretched out. I saw a buckling in the 3rd dimension as hoped.
- Fixed dangling sides in the pop-up pattern bu changing the pattern to avoid this. I also tried adding sideboxes as a more general fix. This kinda worked but restrained the pop-up effect a bit and did not yield as clean results.
- Fixed some bugs in the pop_pattern generic formulation of unit2_axis.
- Made framework for simulating the sheet next to (on top) of a diamond block for contact area measurements.  
- Encounted problem of initial configuration was not really stable. It expands immediately and oscillates back and forth a bit (looking into configuration changes).
- Unstable diamond problems seem to be fixed by having at least 2 unit cells of thickness
- Dealing with: ERROR on proc 0: Neighbor list overflow, boost neigh_modify one (src/npair_full_bin_ghost.cpp:151). 
- Seems like neigh_modify one 10000 solved the issue. Still have to understand what these commands are doing....


#### Week 36 (05/09 - 11/09) - Working from Denmark 
- Fixed the problem with multiple lammps "fix" giving warning: WARNING: One or more atoms are time integrated more than once (src/modify.cpp:292)
- Made python code for calculation minimum distance for each sheet atom for quantification of contact area
- Working on C++ version of above for better performance 
- Problem with sheet getting sucked into the diamond block before even stretching the sheet. A 8% stretch did not change the number of contact particles visible. Trying with Silicium instead (following: https://www.nature.com/articles/nature20135) and stretching harder. 
- In order to use Silicium I need to look into different potentials: using sw for Si-Si, tersoff for C-C and LJ for Si-C.
- For the contact number distance threshold I might need to look into potential equlibrium lengths. 
- Worked on anamorphous Silicon. Parked the problem for now and went with crystal-Si as substrate first, but got the original code from the authors of the previous mentioned articles in the mean time. Remember to go back to this, and remember to acknowledge them. 
- Working on more streamlined framework.
- Added option to have more space in the pop-up pattern
- Made C++ code for calculating contact distance and writing to file which is read and processed in a python script.



#### Week 37 (12/09 - 18/09) - Working from Denmark 
- Added code for measuring normal position relative to sheet: normal buckling.
- Playing around with different potentials. Having SW for Si-Si bonds in substrate, LJ for intemolecular forces in C-Si bonds and Tersoff or Airebo for C-C bonds.
- Measureing contact forces and normal buckling when strething the sheet. Playing around with different starting heights and having the pulling blocks integrated or fixed during relaxation. 
- Stretching the sheet after letting it freely contact with the substrate gives less clear data for the contact area vs stretch. 
- Meeting with Henrik. Make longer simulations of area vs stretch and hysteresis to avoid noise from oscillations. Agreed to try out a force when stretching instead of fix move. Thus we can lower the sheet down in equlibrium distance and then stretch such that the sheet have the change to lift it self up if possible. 
- Next goal is to try difference stretch levels and then drag the whole sheet across the substrate measuring dynamic friction coefficient. If the stretching effefct can lower the coefficient we might be able to make a nanomachine which translates normal force into stretch force and thus perhaps achieve negative friction coefficients (exiting stuff, lets see).
- Proposed pipeline for above study. Make a simulation where the sheet relaxes and falls into contact with the substrate (in vacuum). Then we stretch and export restart files during the process. The restart files can then be used for different stretching starting points for friction measurements. 
- Made updated procedure for stretching and contracting sheet with ouput of stretch pct and forces acting on the pull blocks. Implemented on sheet in vaccum for now for hysteresis plots.
- Looked a hysteresis for sheet in vacuum. Some signs of hysteresis but not conclusive. But definetely seeing buckling effect when stretched. Now trying with sheet and substrate.
- Working on connection to UiO clusters. Made ssh-key. Tried slurm and sbatch for submitting job on bigfacet cluster
- Looked at hysteresis for sheet and substrate: seemingly hysteresis going on for the bottom expansion. In the final part of the contraction the lower quartiles drops compared to starting point. This might be biased by median moving after stretching start also.
- Made longer run on bigfacet for 


#### Week 38 (19/09 - 25/09) - Working from Oslo
- Made a friction procedure (relax, stretch, pause, add normal force, pause, drag sheet along substrate). Now working on friciton force measurement.
- Added more precise way to keep track of stretch pct by measuring min and max of PB
- Added write to file of friction forces 


#### Week 39 (26/09 - 02/10) - Working from Pisa
- Working on details for friciton procedure (virtual spring might not be perfect)
- Reconsidered friction procedure and is going to remake it for mathcing real world system of a nano machine. 
- Made structured pipeline consisting of lammps files: commands with variables for giving procedure, setup of system, the procedure to do (friction simulation for instance).
- Submitted friction simulation with different drag speed of 1, 2, 5, 10, and 50 m/s... Did not run due to LAMMPS issue when running on cluster
- Ran small simulations on local computer and got some results regarding friction forces. Both spring force and group/group contact force between sheet and substrate seems promising. Now I need to run longer simulations and apply a savgol filter to clean up the data and determine static and dynamic friction coefficient if possible.


#### Week 40 (03/10 - 09/10) - Working from Pisa
- Working on determining the problem when running on the cluster (simulation explodes)
- Isolated the problem to the hybrid style potentials. The issue only occours when the Si-atom substrate and C-atom sheet is closer to each other than the AIREBO (C-C interactions) cutt-off
- Working on a procedure for pushing the sheet and substrate together with weaker normal force before applying bigger one (to avoid violent impact) --> Settled on damping 
- Changed parameters for LJ (had used a vary small cuttoff distance)
- Changed how I apply normal force. Before I applied on the whole sheet now I do only apply it on the pullblocks.
- Changed som details on friction procedure simulation
- Done longer simulation locally to see how friction forces spikes. Looks likes the stick-slip motion becomes pretty consistent without any difference in first spike and later ones. No real difference between static and dynamic friction other than taking max and mean force. 
- I have been reading on tribology and gathered some reading material. I have also began to add som quotes and more sections to the article.
- Found out that the AIREBO potential is not compatible with KOKKOS. So instead I should run on CPU (using egil)
- Finally sumbitted a script to the cluster (now CPU) 4 friction simulations: no cut no stretch, no cut stretch, cut no stretch and cut stretch.
- Might need to try out Evens package for running lammps script on the cluster.

#### Week 41 (10/10 - 16/10) - Working from Pisa
- Fixed some issues in lammps script with reulsted in crash on cluster
- Updated output: compute friction force on both inner sheet and pull blocks 
- Updated pipeline for starting simulations with the use of lammps-simulator package (from Even)
- Added SSH compatability to lammps-simulator 
- Result are in for friction coefficient, trends are: (same cut): stretch < no stretch, (stretch):  pop-up-pattern > no cuts
- Ran with 1 m/s drag speed (compared to 5 m/s) which showed same trend except for (cut, stretch) where the friction coefficient (by max force) increased by a factor 4 due to static friction sticking behaviour. Now trying with 0.25 m/s and longer drag length. 
- Also ran with dt = 0.5 fs (compared to 1 fs). This made some noticable changes to friction coefficient (up to 36\% relative divation), but trends remained the same. Although the ordering of value was not the same: (X, 0\%), (√, 0\%), (√, 20\%), (X, 0\%) --> (√, 0\%), (√, 20\%), (X, 0\%), (√, 20\%). So it seems to have enforced the effect from having cuts, while the stretched version still comes in with lowest friction coefficient. 


#### Week 42 (17/10 - 23/10) - Working from Pisa
- Discovered that the sheets ruptures when dragged slow enough (having enough time). This is most likely due to a too high normal force of 160 nN.
- Made new simulation with nvt and lower normal force. Now the arising patterns seems to be (√, 20%) > (√, 0%) > (X, 0%) ~ (X, 20%). The first 'equality' holds for 1 m/s -> 5 m/s, and spring drag (K = 30) -> Fix Move.
- Finished pipeline for sumbitting multiple simulations (combinations of stretch and normal force) for each cut configuration.    
- Checked simulations with Fix move and dt = 0.5 fs which seems to follow the order mentioned in this weeks log. 
- Ran one config multi data simulation and saw some quick curves for (F_N, F_f) at fixed stretch and (stretch, Ff) at fixed FN. With very fex data points we see more or less linear trend for (F_N, F_f). For (stretch, F_f) it starts linear but then flattens/drops at high stretch. I am going to get more data points to further investegate this. 
- Next: Work on coordination number for detecting fractures. 


#### Week 43 (24/10 - 30/10) - Working from Pisa
-  Ran 15 stretch x 10 F_N and got some curves there. The Ff vs. F_N looks mostly linear, but the Ff vs. stretch shows signs of non linearity and both positive and negative slopes. 
- Due to suspicious behaviour when breaking (Henrik's point of view) and the fact that airebo cannot run on GPU I will try the tersoff potential instead. This will hopefully give the opportunity to run more simulations for the machine learning. 
- Implemented Tersoff, but cannot yet run it on GPU due to some unresolved bug. Until then I run on CPU and investigate the differences from AIREBO:
- The order in the great4 runs when sorting for full_sheet drag parallel friction force is more or less the same for the referece simulation. the friction force is in general a bit higher and a bit differently distributed between sheet and PB. The sheet seems more stable and less prone to ruptures. When it ruptures it breaks in larger bits and do not hang in thin strings anymore.
- Implemented a naive procedure to detect ruptures from coordination number. 
- Improved rupture detector and looks promising so far
- Found a way to calculate contact area in lammps (both as \% and per vector value (True/False) by cutoff).


#### Week 44 (31/10 - 06/11) - Working from Denmark
- Improved rupture detector 
- Working getting Tersoff on GPU and making sure that Tersoff results is stable and not to different from the AIREBO results (which looks good so far)
- Added contact bonds to data file
- Updated analysis python methods
- (Mostly spend times on FYS3150 correcting)


#### Week 45 (07/11 - 13/11) - Working from Pisa
- Fixed the GPU tersoff problem (updating lammps on cluster)
- Working on fixing slow GPU speeds (spend a lot of times on the 'modify' category)
- Doing benchmarking of computation speeds with tersoff potential. Possible problem on GPU usage.
- Expanded analysis code, and looked at relations between contact area and friction. At the moment the relation does not seem to be linear, but more results are needed to make this conclusion. 


#### Week 46 (14/11 - 20/11) - Working from Pisa
- Quick summary
- Ran big MULTI with and without stretch but the difference is not clear
- Thus working on good decription on the no stretch case
- This lead to the discovery that the drag length is nt sufficient for the measurements to stabilize
- Now working on setting parameters for the friciton simmulation that is stable
- Working on cummulative top quantile (99\% - 99.9\%) max mean as a more stable way to measure the max friction (static friction).

#### Week 47 (21/11 - 27/11) - Working from Pisa
- Considering only looking at contact area to long computation time for stable result. Solution might be to just ignore max friction since there is no clear sign of static friction. I think contact area and mean friction is more of less equally demanding. 
- Playing around with paramters to see if I can get something that is stable at a reasonable computation time. 
- Trying new (small) multi sim with and without cuts to check out results. 
- Making amorphic silicon to invistigate the effect on the friction measurements. 
- In the mean while I aim to broaden my view on the experimental and numerical results related to graphene friction in order to update my expectation regarding frictional behaviour such as static friction and stick-slip motion. 
- Discovered small problem with the fix ave force on the pull blocks that kept the sheet in an unrelaxed position. FIxed that and had to put int some spring forces in the relax part to keep the sheet in the desired start posistion to avoid any inconsistencies.
- Redooing Baseline test due to non relaxed sheet at zero stretch.
- Trying using gold substrate as this is used in [Study of Nanoscale Friction Behaviors of Graphene on Gold Substrates Using Molecular Dynamics](https://nanoscalereslett.springeropen.com/articles/10.1186/s11671-018-2451-3). 
- Getting more stable results with the fix. Amorph is not the way to go as it is more snesitive and unstable, and even though gold gives more pretty friction profiles it still needs around the same drag distance and is much slower to simulate.
- For the non stretch sheet with 200 nN I observe that the non-bonded atoms aligns with the substrate such that the non-bonded atoms forms lines that stays in space as the sheet moves over the substrate. This is seen on multiple drag speeds. 


#### Week 47 (28/11 - 04/12) - Working from Pisa
- Did my midway presentation.
- I found that low range $F_N$ gives more promising results regarding an increase in friction as a function of stretch for my pop up pattern. This was in the range [0.1, 10] nN. 
- I cannot show any connection to contact area and friction, and if I have to force a connection upon the data I actually see that friction decrease with contact area for my pop up pattern. This might be specific to this pattern and I will wait to see how this behaves for a different pattern. 
- I tested different cut off for contact area with the idea that this might be the course of the above point, but the result was pretty stable an qualitatively the same.

#### Week 48 (05/12 - 11/12) - Working from Pisa
- Beginnning the work on the multi configuration data generation pipeline. 
- Planning to do some stability test to lock in my simulations parameters in the mean while. 
- Made a script for testing stretch range of a cut confiugration in lammps. This includes an in-lammps method for detecting rupture using y-stress and cluster count. I plan to implement this for the friciton simulaiton as well to completely bypass a secondary analysis using coordination numbers. 
- Changed Cdis from 1.42 Å to 1.461 Å as this gave a more stable initial strutcure in practice. During that I searched for optimized tersoff parameters for graphebne since this deviation from theoretical atom spacing was odd. But the optimized parameters I tried did not work well for the cutted configuration so I'm sticking to the orignal parameters for C-C bonds. 
- Implemented the rupture test directly in the friction simulation procedure. This also makes it possible to run the rupture test wihtin the setup_sim.in stretch.in framework. 
- Next step is to implement the opportunity to create the Si-substrate directly in lammps such that I can run rupture test and go directly to a friction simulation with a suitable substrate size. 

#### Week 49 (12/12 - 18/12) - Working from Pisa
- Implemented substrate build in lammps, keeping oportunity to providing the substrate through a txt file. 
- Implemented refined definition of rupture stretch in rupture test
- Cleaned up in computes and fixes.
- Working on pipeline for configuration folder (numpy files) -> data.
- Doing some test on bigger sheets. So far nothing alarming og surprising. Not really a big difference in stability other than it might stabilize a bit faster than small sheets.



## Things to remember 

### Words
- Out-of-plane buckling

### General stuff
- Active learning for generating datasets entries while training


### From https://www.nature.com/articles/nature20135. 
- Use this as reference for material choice and friction "experiment" set up. 
 - It has been shown experimentally that monolayer graphene exhibits higher friction than multilayer graphene and graphite, and that this friction increases with continued sliding, but the mechanism behind this remains subject to debate.
 -  It has long been conjectured that the true contact area between two rough bodies controls interfacial friction1. The true contact area, defined for example by the number of atoms within the range of interatomic forces, is difficult to visualize directly but characterizes the quantity of contact.
 - The tip–graphene contact area is taken
to be ms, where m is the number of graphene atoms that are in intimate contact with the tip atoms and s (2.77 Å2 per atom) is the atomic area of graphene (see Methods for details).
- Talked to Henrik and we decided to change the diamond to Silicon. I am now following the set up shown in the article: https://www.nature.com/articles/nature20135#Sec8.
- Thus I am working on the first part quenching liquid silicon to form the substrate (both for area vs stretch) but also as the layer under the graphene when press a tip into the sheet.
- Tune LJ potential to meet real adhesion energy and adhesion force. 