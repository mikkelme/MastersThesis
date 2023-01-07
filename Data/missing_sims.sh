#!/bin/bash


pwd=$(pwd)

shopt -s nullglob
for s in $1/* ; do
   for d in $s/job* ; do
	if compgen -G "$d/*.data" > /dev/null; then
		:
	else
		echo $d
		cd $d
		JOB=$( find . -maxdepth 1 -type f -name "*.sh") 
		SLURM=$( find . -maxdepth 1 -type f -name "*.out")
		
		#echo "$JOB"
		#echo "$SLURM"
	
		
		if [ ! -z $SLURM ] ; then
			rm $SLURM
		fi

		sbatch $JOB
		cd $pwd
                
	fi
   done
done
echo "Done searching in $1"
