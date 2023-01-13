#!/bin/bash
# Bash script for fetching newest data to a folder custer using rsync

ssh=egil
# FLAGS=(--include '*/' --include '*.txt' --include '*.npy' --include '*.png')
FLAGS=(--include '*/' --include '*.txt')


if [ $# -eq 0 ] 
then # No folder provided
    echo "No folder provided"

elif [ $# -eq 1 ] 
then # if num files == 1
    echo Updating: $1 via $ssh
    rsync -av --progress "${FLAGS[@]}" --exclude '*' ${ssh}:$1/ ./$1

else # if num files > 1
    for ((i=1; i<=$#; i++))
    do
        echo Updating: ${!i} via $ssh
        rsync -av --progress "${FLAGS[@]}" --exclude '*' ${ssh}:${!i} ./ 
    done
fi
