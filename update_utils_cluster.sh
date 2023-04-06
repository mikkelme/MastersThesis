#!/bin/bash
# Bash script for transfering python script to cluster

FLAGS=(--include '*/' --include '*.py' --exclude '*' ) # Include sub dirs
# FLAGS=(--include '*.py' --exclude '*' )

array=( "/graphene_sheet" "./analysis" "config_builder" "produce_figures" )

if [ $# -eq 0 ] 
then # No ssh host provided
    ssh=bigfacet
    echo "Using default ssh:${ssh}"
    for folder in "${array[@]}"
    do
        echo "${folder}"
        rsync -av --progress "${FLAGS[@]}" ./${folder} ${ssh}:./
    done

elif [ $# -eq 1 ] 
then # ssh host provided
    ssh=$1
    echo Using provided ssh:$1
    for folder in "${array[@]}"
    do
        echo "${folder}"
        rsync -av --progress "${FLAGS[@]}" ./${folder} ${ssh}:./
    done

else # multiple ssh host providedif num files > 1
    echo Error: Multiple ssh host provided:
    for ((i=1; i<=$#; i++))
    do
        echo ssh:${!i} 
    done
fi


