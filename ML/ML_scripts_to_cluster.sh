#!/bin/bash
# Bash script for transfering python script to cluster

# FLAGS=(--include '*/' --include '*.py' --exclude '*' ) # Include sub dirs
FLAGS=(--include '*.py' --exclude '*' )

if [ $# -eq 0 ] 
then # No ssh host provided
    ssh=bigfacet
    echo "Using default ssh:${ssh}"
    rsync -av --progress "${FLAGS[@]}" ./ ${ssh}:ML/

elif [ $# -eq 1 ] 
then # ssh host provided
    ssh=$1
    echo Using provided ssh:$1
    rsync -av --progress "${FLAGS[@]}" ./ ${ssh}:ML/

else # multiple ssh host providedif num files > 1
    echo Error: Multiple ssh host provided:
    for ((i=1; i<=$#; i++))
    do
        echo ssh:${!i} 
    done
fi


# Include plot set from parent folder as well
rsync -av --progress /Users/mikkelme/Documents/Github/MastersThesis/plot_set.py ${ssh}:./
