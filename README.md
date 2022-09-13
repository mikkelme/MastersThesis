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






## Things to remember 

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