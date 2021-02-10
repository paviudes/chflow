#!/bin/bash

timestamps=("biased_pauli_Steane" "biased_pauli_cyclic")

cluster="$2"

host=$(hostname)

if [[ $host == *"paviws"* ]]; then
	local_user=${USER}
	# email=pavithran.sridhar@gmail.com
	outdir="/Users/pavi/Documents/chbank"
	chflowdir="/Users/pavi/Documents/rclearn/chflow"
	sed_prepend=' '
	
elif [[ $host == "pavitp" ]]; then
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

if [[ "$1" == "zip" ]]; then
	cd ${outdir}
	printf "\033[2m"
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "zipping ${ts}"
		tar -zcvf ${ts}.tar.gz ${ts}
		# move the input and schedule files into the output folder
		cp ${chflowdir}/input/${ts}.txt ${ts}/input/
		cp ${chflowdir}/input/schedule_${ts}.txt ${ts}/input/
	done
	cd ${chflowdir}
	printf "\033[0m"

elif [[ "$1" == "from_cluster" ]]; then
	cd ${outdir}
	printf "\033[2m"
	echo "Bringing simulation from ${cluster}"

	# unzip the individual datasets.
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}

		scp -r ${local_user}@${cluster}.computecanada.ca:/project/def-jemerson/chbank/${ts}.tar.gz ${outdir}
	
		echo "Setting up ${ts}"
		trash ${ts}.tar.gz
		trash ${ts}
		tar -xvf ${ts}.tar.gz
		
		#### Copying input files to chflow
		echo "Copying input and schedule files"
		cp ${ts}/input/${ts}.txt ${chflowdir}/input/
		echo "Copying schedule_${ts}.txt from data"
		cp ${ts}/input/schedule_${ts}.txt ${chflowdir}/input/
		
		#### Prepare output directory after moving from cluster.
		echo "/project/def-jemerson/chbank WITH ${outdir} IN input/${ts}.txt"
		replace "\/project\/def-jemerson\/chbank" ${outdir//\//\\\/} ${chflowdir}/input/${ts}.txt
		
		echo "xxxxxxx"
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