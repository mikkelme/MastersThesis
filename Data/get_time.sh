#!/bin/bash
# Print wall time of simulations in folder
# provided as command line argument


if [ $# -eq 0 ] ; then # No folder provided
    echo "No folder provided"

elif [ $# -eq 1 ] ; then # if num files == 1
    	shopt -s nullglob
	for s in $1/* ; do
   		for d in $s/job* ; do
        		if compgen -G "$d/*drag.data" > /dev/null; then
                		SLURM=$( find $d/ -maxdepth 1 -type f -name "*.out")
                		tag=$( tail -n 1 $SLURM )
                		echo "$tag"
        		fi
   		done
	done

else # if num files > 1
    echo "Cannot provide multiple folders"
   
fi



#shopt -s nullglob
#for s in ./CONFIGS/cut_sizes/conf_6/stretch* ; do
#   for d in $s/job* ; do
#	if compgen -G "$d/*drag.data" > /dev/null; then
#		SLURM=$( find $d/ -maxdepth 1 -type f -name "*.out")
#		tag=$( tail -n 1 $SLURM )
#		echo "$tag"
#	fi
#   done
#done

