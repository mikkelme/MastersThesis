#!/bin/bash
# Collect files from ssh cluster
# Input files in command line divided by space

if [ $# -eq 1 ] 
then # if num files == 1
    scp -r bigfacet:/home/users/mikkelme/$1 ./inbox 

else # if num files > 1
    string="$1"
    for ((i=2; i<=$#; i++))
    do 
        string=${string}",${!i}"
    done

    scp -r bigfacet:/home/users/mikkelme/{$string} ./inbox 
    echo $string
fi
