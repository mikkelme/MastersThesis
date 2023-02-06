#!/bin/bash
# Search dir to find runs which fail to run and resubmit them


ext="Ff.txt" 	# Target extension to determine whether job has startedi
pwd=$(pwd) 	# Initial pwd
execute=false 	# Whether to rm old slurm and sbatch job

dir=$1 # Search dir for unstarted jobs 
for i in $(find $dir -name "job.sh") # Look for job-files in dir and subdirs 
do
	parentdir="$(dirname "$i")"
	subdircount=`find $parentdir/ -maxdepth 1 -type d | wc -l`

	if [ ! $subdircount -ge 2 ] # if subdir contains no other childdirs
	then 
		if [ ! -f $parentdir/*$ext ] # if subdir does not contain target extension
		then 
			echo $parentdir
			if $execute
			then
               			cd $parentdir
               			JOB=$( find . -maxdepth 1 -type f -name "*.sh")
               			SLURM=$( find . -maxdepth 1 -type f -name "*.out")

                		if [ ! -z $SLURM ] # Delete old slum if existing
				then
                			rm $SLURM
                		fi

               			sbatch $JOB
              			cd $pwd

				
			fi
		fi
	fi
done



#pwd=$(pwd)

#shopt -s nullglob
#for d in $1/* ; do
  #     if [ -f $d/*Ff.txt ] ; then
 #  		echo "yes"
#	else
#		echo "no"
#	fi 	
	#if compgen -G "$d/*.Ff.txt" > /dev/null; then :
	#	echo "no"
	#else
	#	echo $d
	#fi

	
 #  for d in $s/* ; do
#	   echo $d
 
#	if compgen -G "$d/*.data" > /dev/null; then
#		:
#	else
#		echo $d
#		cd $d
#		JOB=$( find . -maxdepth 1 -type f -name "*.sh") 
#		SLURM=$( find . -maxdepth 1 -type f -name "*.out")
		
		
		
		#if [ ! -z $SLURM ] ; then
		#	rm $SLURM
		#fi

		#sbatch $JOB
#		cd $pwd
                
#	fi
   #done
#done
#echo "Done searching in $1"
