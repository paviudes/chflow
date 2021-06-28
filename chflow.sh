#!/bin/bash

timestamps=("pcorr_batch3_Steane_l2_00")

cluster="$2"

host=$(hostname)

if [[ $host == *"paviws"* ]]; then
	local_user=${USER}
	# email=pavithran.sridhar@gmail.com
	outdir="/Users/pavi/Documents/chbank"
	chflowdir="/Users/pavi/Documents/rclearn/chflow"
	sed_prepend=' '
	
elif [[ $host == "pavitp" ]]; then
	local_user=${USER}
	outdir="/home/pavi/Documents/chbank"
	chflowdir="/home/pavi/Documents/chflow"
	cores=$(nproc --all)
	
elif [[ $host == "oem-ThinkPad-X1-Carbon-Gen-8" ]]; then
	if [[ -z ${local_user} ]]; then
		local_user=${USER}
	fi
	# email=a77jain@gmail.com
	outdir="/home/oem/Documents/chbank"
	chflowdir="/home/oem/Desktop/Research_PhD/chflow"

else
	outdir="/project/def-jemerson/chbank"
	chflowdir="/project/def-jemerson/${USER}/chflow"
fi

replace() {
	# run sed to replace a substring with another.
	if [[ -n ${sed_prepend} ]]; then
		sed -i "${sed_prepend}" "s/$1/$2/g" $3
	else
		sed -i "s/$1/$2/g" $3
	fi
}

fastdelete() {
	# Delete efficiently using perl
	# https://unix.stackexchange.com/questions/37329/efficiently-delete-large-directory-containing-thousands-of-files
	cd $1
	perl -e "for(<*>){((stat)[9]<(unlink))}"
	cd $chflowdir
}

if [[ "$1" == "zip" ]]; then
	cd ${outdir}
	printf "\033[2m"
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "zipping ${ts}"
		# move the input and schedule files into the output folder
		cp ${chflowdir}/input/${ts}.txt ${ts}/input/
		cp ${chflowdir}/input/schedule_${ts}.txt ${ts}/input/
		tar -zcvf ${ts}.tar.gz ${ts}
	done
	cd ${chflowdir}
	printf "\033[0m"

elif [[ "$1" == "from_cluster" ]]; then
	cd ${outdir}
	printf "\033[2m"
	echo "Bringing simulation data from ${cluster} to ${outdir}"
	# unzip the individual datasets.
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}

		trash ${outdir}/${ts}.tar.gz
		
		echo "scp ${local_user}@${cluster}.computecanada.ca:/project/def-jemerson/chbank/${ts}.tar.gz ${outdir}"
		scp ${local_user}@${cluster}.computecanada.ca:/project/def-jemerson/chbank/${ts}.tar.gz .
	
		echo "Setting up ${ts}"
		trash ${ts}
		tar -xvf ${ts}.tar.gz
		
		#### Copying input files to chflow
		echo "Copying input and schedule files"
		cp ${ts}/input/${ts}.txt ${chflowdir}/input/
		cp ${ts}/input/schedule_${ts}.txt ${chflowdir}/input/
		
		#### Prepare output directory after moving from cluster.
		echo "REPLACING /project/def-jemerson/chbank WITH ${outdir} IN ${chflowdir}/input/${ts}.txt"
		replace "\/project\/def-jemerson\/chbank" ${outdir//\//\\\/} ${chflowdir}/input/${ts}.txt
		
		echo "xxxxxxx"
	done
	printf "\033[0m"
	cd ${chflowdir}

elif [[ "$1" == "delete" ]]; then
	printf "\033[2m"
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		if [ -d ${outdir}/${ts}/physical ]; then
			echo "removing ${outdir}/${ts}/physical/*"
			fastdelete ${outdir}/${ts}/physical/
		else
			echo "No physical found in ${outdir}/${ts}."
		fi
		if [ -d ${outdir}/${ts}/channels ]; then
			echo "removing ${outdir}/${ts}/channels/*"
			fastdelete ${outdir}/${ts}/channels/
		else
			echo "No channels found in ${outdir}/${ts}."
		fi
		if [ -d ${outdir}/${ts}/metrics ]; then
			echo "removing ${outdir}/${ts}/metrics/*"
			fastdelete ${outdir}/${ts}/metrics/
		else
			echo "No metrics found in ${outdir}/${ts}."
		fi
		if [ -d "${outdir}/${ts}/results" ]; then
			echo "removing ${outdir}/${ts}/results/*"
			rm ${outdir}/${ts}/results/*.npy
		else
			echo "No results found in ${outdir}/${ts}."
		fi
		echo "-----"
	done
	printf "\033[0m"


elif [[ "$1" == "chmod" ]]; then
	cd ${outdir}
	printf "\033[2m"
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "Changing permissions for ${ts}"
		chmod -R 777 ${ts}
	done
	printf "\033[0m"
	cd ${chflowdir}
	
else
	cd src
	python chflow.py "$@"
	cd ..
fi
