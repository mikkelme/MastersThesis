## Master's thesis - Predicting Frictional Properties of Graphene Kirigami Using Molecular Dynamics and Neural Networks

This repository contains all related work to my master's thesis for the degree in Computational Science: Materials Science at the University of Oslo.

### Abstract
Various theoretical models and experimental results propose different governing mechanisms for friction at the nanoscale. We consider a graphene sheet modified with Kirigami-inspired cuts and under the influence of strain. Prior research has demonstrated that this system exhibits out-of-plane buckling, which may cause a decrease in contact area when sliding on a substrate. According to asperity theory, such a decrease in contact area is expected to reduce friction. However, to the best of our knowledge, no previous studies have investigated the frictional behavior of a nanoscale Kirigami graphene sheet subjected to strain. Here we show that specific Kirigami designs yield a non-linear dependency between kinetic friction and the strain of the sheet. Using molecular dynamics, we have found a non-monotonic increase in friction with strain. We found that the friction-strain relationship does not show any clear dependency on contact area which contradicts asperity theory. Our findings suggest that the effect is associated with the out-of-plane buckling of the graphene sheet and we attribute this to a commensurability effect. By mimicking a load-strain coupling through tension, we were able to utilize this effect to demonstrate a negative friction coefficient on the order of −0.3 for loads in the range of a few nN. In addition, we have attempted to use machine learning to capture the relationship between Kirigami designs, load, and strain, with the objective of performing an accelerated search for new designs. Although this approach yielded some promising results, we conclude that further improvements to the dataset are necessary in order to develop a reliable model. We anticipate our findings to be a starting point for further investigations of the underlying mechanism for the frictional behavior of a Kirigami sheet. For instance, the commensurability hypothesis could be examined by varying the sliding angle in simulations. We propose to use an active learning strategy to extend the dataset for the use of machine learning to assist these investigations. If successful, further studies can be done on the method of inverse design. In summary, our findings suggest that the application of nanoscale Kirigami can be promising for developing novel friction-control strategies.

### Repository structure
The thesis is found in [article/thesis.pdf](article/thesis.pdf) with supporting figures in  [article/figures](article/figures). 
The main code developed can be found at the following locations.
- LAMMPS scripts for the friction simulation: [friction_simulation](friction_simulation). 
- Generating Kirigami patterns: [config_builder](config_builder) and [graphene_sheet](graphene_sheet). 
- Running multiple simulations (ssh to cluster): [simulation_manager](simulation_manager) and [data_pipeline](data_pipeline). 
- Data analysis and producing figures: [analysis](analysis) and [produce_figures](produce_figures).
- Machine learning: [ML](ML).