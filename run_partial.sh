#!/bin/bash
host=$(hostname)
# echo "Host: $host"
cluster="$2"

if [[ -n ${cluster} ]]; then
	cores=40 # 48 for cedar, 40 for beluga and 32 for graham
	local_user=${USER}
	if [[ $local_user == *"pavi"* ]]; then
		email=pavithran.sridhar@gmail.com
	elif [[ $local_user == *"a77jain"* ]]; then
		email=2003adityajain@gmail.com
	fi
fi

if [[ $host == *"paviws"* ]]; then
	local_user=${USER}
	# email=pavithran.sridhar@gmail.com
	cores=$(sysctl -n hw.ncpu)
	outdir="/Users/pavi/Documents/chbank"
	chflowdir="/Users/pavi/Documents/rclearn/chflow"
	report_dir="/Users/pavi/OneDrive\ -\ University\ of\ Waterloo/chbank/Nov4"
	sed_prepend=' '
	# CPTP
	pavi_ws_cptp_level3=("pavi_ws_cptp_l3_00" "pavi_ws_cptp_l3_01" "pavi_ws_cptp_l3_02" "pavi_ws_cptp_l3_03" "pavi_ws_cptp_l3_04" "pavi_ws_cptp_l3_05" "pavi_ws_cptp_l3_06" "pavi_ws_cptp_l3_07" "pavi_ws_cptp_l3_08" "pavi_ws_cptp_l3_09" "pavi_ws_cptp_l3_10" "pavi_ws_cptp_l3_11")
	alphas_pavi=(0 0.0002 0.0004 0.0008 0.0016 0.0032 0.0063 0.0126 0.0251 0.0501 0.1 1)
	# Command to rename files
	# find . -maxdepth 1 -type d -name "pavi_beluga_cptp_l3_08_12_2020_*" -exec bash -c 'mv $0 ${0/cptp_l3_08_12_2020/cptp}' {} \;

elif [[ $host == "pavitp" ]]; then
	outdir="/home/pavi/Documents/chbank/aditya_cptp/cptp_l3"
	chflowdir="/home/pavi/Documents/chflow"
	cores=$(nproc --all)
	
elif [[ $host == "oem-ThinkPad-X1-Carbon-Gen-8" ]]; then
	if [[ -z ${local_user} ]]; then
		local_user=${USER}
	fi
	# email=a77jain@gmail.com
	outdir="/home/oem/Documents/chbank"
	chflowdir="/home/oem/Desktop/Research_PhD/chflow"
	cores=$(nproc --all)
	ising_level3=("ising_l3_08_12_2020_00" "ising_l3_08_12_2020_01" "ising_l3_08_12_2020_02" "ising_l3_08_12_2020_03" "ising_l3_08_12_2020_04" "ising_l3_08_12_2020_05" "ising_l3_08_12_2020_06" "ising_l3_08_12_2020_07")
	npcorr_level2=("npcorr_l2_08_12_2020_00" "npcorr_l2_08_12_2020_01" "npcorr_l2_08_12_2020_02" "npcorr_l2_08_12_2020_03" "npcorr_l2_08_12_2020_04" "npcorr_l2_08_12_2020_05" "npcorr_l2_08_12_2020_06" "npcorr_l2_08_12_2020_07")
	pcorr_level2=("pcorr_l2_08_12_2020_00" "pcorr_l2_08_12_2020_01" "pcorr_l2_08_12_2020_02" "pcorr_l2_08_12_2020_03" "pcorr_l2_08_12_2020_04" "pcorr_l2_08_12_2020_05" "pcorr_l2_08_12_2020_06" "pcorr_l2_08_12_2020_07")
	pcorr_level3=("pcorr_l3_08_12_2020_00" "pcorr_l3_08_12_2020_01" "pcorr_l3_08_12_2020_02" "pcorr_l3_08_12_2020_03" "pcorr_l3_08_12_2020_04" "pcorr_l3_08_12_2020_05" "pcorr_l3_08_12_2020_06" "pcorr_l3_08_12_2020_07")
	# cptp_level2=("cptp_l2_08_12_2020_00" "cptp_l2_08_12_2020_01" "cptp_l2_08_12_2020_02" "cptp_l2_08_12_2020_03" "cptp_l2_08_12_2020_04" "cptp_l2_08_12_2020_05" "cptp_l2_08_12_2020_06" "cptp_l2_08_12_2020_07")
	cptp_level2=("cptp_l2_24_12_2020_00" "cptp_l2_24_12_2020_01" "cptp_l2_24_12_2020_02" "cptp_l2_24_12_2020_03" "cptp_l2_24_12_2020_04" "cptp_l2_24_12_2020_05" "cptp_l2_24_12_2020_06" "cptp_l2_24_12_2020_07")
	# alphas=(0 0.00013 0.00027 0.00093 0.00368 0.00391 0.00415 0.00678)
	alphas=(0 0.0003 0.001 0.005 0.01 0.05 0.1 1)

else
	outdir="/project/def-jemerson/chbank"
	chflowdir="/project/def-jemerson/${USER}/chflow"
fi

if [[ -n ${cluster} ]]; then
	cores=40 # 48 for cedar, 40 for beluga and 32 for graham
	email=pavithran.sridhar@gmail.com
	## Timestamps
	# ISING
	pavi_beluga_ising_level3=("pavi_beluga_ising_l3_00" "pavi_beluga_ising_l3_01" "pavi_beluga_ising_l3_02" "pavi_beluga_ising_l3_03" "pavi_beluga_ising_l3_04" "pavi_beluga_ising_l3_05" "pavi_beluga_ising_l3_06" "pavi_beluga_ising_l3_07" "pavi_beluga_ising_l3_08" "pavi_beluga_ising_l3_09" "pavi_beluga_ising_l3_10" "pavi_beluga_ising_l3_11" "pavi_beluga_ising_l3_12" "pavi_beluga_ising_l3_13" "pavi_beluga_ising_l3_14" "pavi_beluga_ising_l3_15" "pavi_beluga_ising_l3_16" "pavi_beluga_ising_l3_17" "pavi_beluga_ising_l3_18" "pavi_beluga_ising_l3_19")
	# CPTP
	pavi_beluga_cptp_level3=("pavi_beluga_cptp_l3_00" "pavi_beluga_cptp_l3_01" "pavi_beluga_cptp_l3_02" "pavi_beluga_cptp_l3_03" "pavi_beluga_cptp_l3_04" "pavi_beluga_cptp_l3_05" "pavi_beluga_cptp_l3_06" "pavi_beluga_cptp_l3_07" "pavi_beluga_cptp_l3_08" "pavi_beluga_cptp_l3_09" "pavi_beluga_cptp_l3_10" "pavi_beluga_cptp_l3_11" "pavi_beluga_cptp_l3_12" "pavi_beluga_cptp_l3_13" "pavi_beluga_cptp_l3_14" "pavi_beluga_cptp_l3_15" "pavi_beluga_cptp_l3_16" "pavi_beluga_cptp_l3_17" "pavi_beluga_cptp_l3_18" "pavi_beluga_cptp_l3_19" "pavi_beluga_cptp_l3_20" "pavi_beluga_cptp_l3_21" "pavi_beluga_cptp_l3_22" "pavi_beluga_cptp_l3_23" "pavi_beluga_cptp_l3_24" "pavi_beluga_cptp_l3_25" "pavi_beluga_cptp_l3_26" "pavi_beluga_cptp_l3_27" "pavi_beluga_cptp_l3_28" "pavi_beluga_cptp_l3_29" "pavi_beluga_cptp_l3_30" "pavi_beluga_cptp_l3_31" "pavi_beluga_cptp_l3_32" "pavi_beluga_cptp_l3_33" "pavi_beluga_cptp_l3_34" "pavi_beluga_cptp_l3_35" "pavi_beluga_cptp_l3_36" "pavi_beluga_cptp_l3_37")
    aditya_beluga_cptp_level3=("aditya_beluga_cptp_l3_00" "aditya_beluga_cptp_l3_01" "aditya_beluga_cptp_l3_02" "aditya_beluga_cptp_l3_03" "aditya_beluga_cptp_l3_04" "aditya_beluga_cptp_l3_05" "aditya_beluga_cptp_l3_06" "aditya_beluga_cptp_l3_07" "aditya_beluga_cptp_l3_08" "aditya_beluga_cptp_l3_09" "aditya_beluga_cptp_l3_10" "aditya_beluga_cptp_l3_11" "aditya_beluga_cptp_l3_12" "aditya_beluga_cptp_l3_13" "aditya_beluga_cptp_l3_14" "aditya_beluga_cptp_l3_15" "aditya_beluga_cptp_l3_16" "aditya_beluga_cptp_l3_17" "aditya_beluga_cptp_l3_18" "aditya_beluga_cptp_l3_19" "aditya_beluga_cptp_l3_20" "aditya_beluga_cptp_l3_21" "aditya_beluga_cptp_l3_22" "aditya_beluga_cptp_l3_23" "aditya_beluga_cptp_l3_24" "aditya_beluga_cptp_l3_25" "aditya_beluga_cptp_l3_26" "aditya_beluga_cptp_l3_27" "aditya_beluga_cptp_l3_28" "aditya_beluga_cptp_l3_29" "aditya_beluga_cptp_l3_30" "aditya_beluga_cptp_l3_31" "aditya_beluga_cptp_l3_32" "aditya_beluga_cptp_l3_33" "aditya_beluga_cptp_l3_34" "aditya_beluga_cptp_l3_35" "aditya_beluga_cptp_l3_36" "aditya_beluga_cptp_l3_37")
    aditya_beluga_cptp_level2=("aditya_beluga_cptp_l2_00" "aditya_beluga_cptp_l2_01" "aditya_beluga_cptp_l2_02" "aditya_beluga_cptp_l2_03" "aditya_beluga_cptp_l2_04" "aditya_beluga_cptp_l2_05" "aditya_beluga_cptp_l2_06" "aditya_beluga_cptp_l2_07" "aditya_beluga_cptp_l2_08" "aditya_beluga_cptp_l2_09" "aditya_beluga_cptp_l2_10" "aditya_beluga_cptp_l2_11" "aditya_beluga_cptp_l2_12" "aditya_beluga_cptp_l2_13" "aditya_beluga_cptp_l2_14" "aditya_beluga_cptp_l2_15" "aditya_beluga_cptp_l2_16" "aditya_beluga_cptp_l2_17" "aditya_beluga_cptp_l2_18" "aditya_beluga_cptp_l2_19" "aditya_beluga_cptp_l2_20" "aditya_beluga_cptp_l2_21" "aditya_beluga_cptp_l2_22" "aditya_beluga_cptp_l2_23" "aditya_beluga_cptp_l2_24" "aditya_beluga_cptp_l2_25" "aditya_beluga_cptp_l2_26" "aditya_beluga_cptp_l2_27" "aditya_beluga_cptp_l2_28" "aditya_beluga_cptp_l2_29" "aditya_beluga_cptp_l2_30" "aditya_beluga_cptp_l2_31" "aditya_beluga_cptp_l2_32" "aditya_beluga_cptp_l2_33" "aditya_beluga_cptp_l2_34" "aditya_beluga_cptp_l2_35" "aditya_beluga_cptp_l2_36" "aditya_beluga_cptp_l2_37")
    
    # aditya_cptp_level2=("cptp_l2_24_12_2020_00" "cptp_l2_24_12_2020_01" "cptp_l2_24_12_2020_02" "cptp_l2_24_12_2020_03" "cptp_l2_24_12_2020_04" "cptp_l2_24_12_2020_06" "cptp_l2_24_12_2020_07" "cptp_l2_24_12_2020_08" "cptp_l2_24_12_2020_10" "cptp_l2_24_12_2020_11" "cptp_l2_24_12_2020_12" "cptp_l2_24_12_2020_13" "cptp_l2_24_12_2020_14" "cptp_l2_24_12_2020_15" "cptp_l2_24_12_2020_16" "cptp_l2_24_12_2020_17" "cptp_l2_24_12_2020_18" "cptp_l2_24_12_2020_19" "cptp_l2_24_12_2020_20" "cptp_l2_24_12_2020_21" "cptp_l2_24_12_2020_22" "cptp_l2_24_12_2020_23" "cptp_l2_24_12_2020_24" "cptp_l2_24_12_2020_25" "cptp_l2_24_12_2020_26" "cptp_l2_24_12_2020_27" "cptp_l2_24_12_2020_28" "cptp_l2_24_12_2020_29" "cptp_l2_24_12_2020_30" "cptp_l2_24_12_2020_31" "cptp_l2_24_12_2020_32" "cptp_l2_24_12_2020_33" "cptp_l2_24_12_2020_34" "cptp_l2_24_12_2020_35" "cptp_l2_24_12_2020_36" "cptp_l2_24_12_2020_37")
    ## Alphas
    # pavi
    alphas_pavi=(0 0.0001 0.00012 0.00015 0.00018 0.00022 0.00027 0.00033 0.0004 0.00048 0.00059 0.00072 0.00088 0.00107 0.0013 0.00158 0.00193 0.00235 0.00287 0.00349 0.00425 0.00518 0.00631 0.00769 0.00936 0.01141 0.01389 0.01693 0.02062 0.02512 0.0306 0.03728 0.04541 0.05532 0.06739 0.08209 0.1 1)
    # aditya
    alphas_aditya=(0 0.0001 0.00012 0.00015 0.00018 0.00022 0.00027 0.00033 0.0004 0.00048 0.00059 0.00072 0.00088 0.00107 0.0013 0.00158 0.00193 0.00235 0.00287 0.00349 0.00425 0.00518 0.00631 0.00769 0.00936 0.01141 0.01389 0.01693 0.02062 0.02512 0.0306 0.03728 0.04541 0.05532 0.06739 0.08209 0.1 1)

fi


rerun() {
	echo "removing ${outdir}/$1/channels/*"
	if [ -d ${outdir}/$1/channels ]; then
		rm ${outdir}/$1/channels/*
	else
		echo "No channels found."
	fi
	if [ -d ${outdir}/$1/metrics ]; then
		rm ${outdir}/$1/metrics/*
	else
		echo "No metrics found."
	fi
	if [ -d "${outdir}/$1/results" ]; then
		rm ${outdir}/$1/results/*.npy
	else
		echo "No results found."
	fi

	# rm ${outdir}/$1/physical/*.npy
	echo "Preparing $1"
	echo "$1" >> input/partial_decoders_$2.txt
}

replace() {
	# run sed to replace a substring with another.
	if [[ -n ${sed_prepend} ]]; then
		sed -i "${sed_prepend}" "s/$1/$2/g" $3
	else
		sed -i "s/$1/$2/g" $3
	fi
}

usage() {
	echo -e "\033[1mUsage: ./run_partial.sh <command> [<cluster>]\033[0m"
	printf "\033[2m"
	echo -e "where, command can be one of"
	echo -e "generate, overwrite, zip, from_cluster, pmetrics, plot, schedule_copy, filter, gdrive, archive, chmod"
	printf "\033[0m"
}

timestamps=("${aditya_beluga_cptp_level2[@]}")
alphas=("${alphas_aditya[@]}")
log=aditya_beluga_cptp_level2
refts=${timestamps[0]}

if [[ "$1" == "overwrite" ]]; then
	rm input/partial_decoders_${log}.txt
	printf "\033[2m"
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		rerun ${ts} ${log}
		echo "xxxxxxx"
	done
	
	if [[ $host == *"paviws"* ]]; then
		echo "Run the following command."
		echo -e "\033[4mparallel --joblog partial_decoders_${log}.log --jobs ${cores} ./chflow.sh {1} :::: input/partial_decoders_${log}.txt\033[0m"
	
	elif [[ $host == "oem-ThinkPad-X1-Carbon-Gen-8" ]]; then
		echo "Run the following command."
		echo -e "parallel --joblog partial_decoders_${log}.log --jobs ${cores} ./chflow.sh {1} :::: input/partial_decoders_${log}.txt"
	
	else
		mkdir -p input/${cluster}
		rm input/${cluster}/partial_decoders_${log}.sh
		sbcmds=("#!/bin/bash" "#SBATCH --account=def-jemerson" "#SBATCH --begin=now" "#SBATCH --nodes=1" "#SBATCH --time=05:00:00" "#SBATCH --ntasks-per-node=${cores}" "#SBATCH -o /project/def-jemerson/chbank/${USER}_${log}_output.o" "#SBATCH -e /project/def-jemerson/chbank/${USER}_${log}_errors.o" "#SBATCH --mail-type=ALL" "#SBATCH --mail-user=${email}")
		for (( s=0; s<${#sbcmds[@]}; ++s )); do
			echo "${sbcmds[s]}" >> input/${cluster}/partial_decoders_${log}.sh
		done
		echo "module load intel python scipy-stack" >> input/${cluster}/partial_decoders_${log}.sh
		echo "cd /project/def-jemerson/${USER}/chflow" >> input/${cluster}/partial_decoders_${log}.sh
		echo "start=\$(date +%s)"
		echo "parallel --joblog partial_decoders_${log}.log ./chflow.sh {1} :::: input/partial_decoders_${log}.txt" >> input/${cluster}/partial_decoders_${log}.sh
		echo "end=\$(date +%s)"
		echo "runtime=\$((end/3600-start/3600))"
		
		# Prepare a summary email
		touch input/summary.txt
		echo "The following job was completed on ${cluster} in ${runtime} hours." >> input/summary.txt
		echo "Job name: ${log}" >> input/summary.txt
		printf -v joined_timestamps '%s, ' "${timestamps[@]:0}"
		echo "Time stamps: ${joined_timestamps%?}" >> input/summary.txt
		printf -v joined_alphas '%s, ' "${alphas[@]:0}"
		echo "Alphas: ${alphas%?}" >> input/summary.txt
		echo "User: ${local_user}" >> input/summary.txt
		echo "Host: ${host}" >> input/summary.txt
		echo "Date: $(date)" >> input/summary.txt
		cat input/summary.txt | mail -s "[${cluster}] ${log} done" ${email}
		rm input/summary.txt

		echo "xxxxxxx"
		
		echo "Run the following command."
		echo "sbatch input/${cluster}/partial_decoders_${log}.sh"
	fi
	printf "\033[0m"

elif [[ "$1" == "schedule_copy" ]]; then
	printf "\033[2m"
	for (( t=1; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "removing ./input/schedule_${ts}.txt"
		rm ./input/schedule_${ts}.txt
		echo "copying ./input/schedule_${timestamps[0]}.txt ./input/schedule_${ts}.txt"
		cp ./input/schedule_${timestamps[0]}.txt ./input/schedule_${ts}.txt
		echo "xxxxxxx"
	done
	printf "\033[0m"

elif [[ "$1" == "generate" ]]; then
	refalpha=${alphas[0]}
	printf "\033[2m"
	echo "sbload ${refts}" > input/temp.txt
	for (( t=1; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		alpha=${alphas[t]}
		echo "alpha = ${alpha}"
		echo "removing ${outdir}/${ts}/physical/*"
		rm ${outdir}/${ts}/physical/*
		# echo "sbtwirl" >> input/temp.txt
		echo "submit ${ts}" >> input/temp.txt
	done
	printf "\033[0m"

	echo "quit" >> input/temp.txt
	./chflow.sh -- temp.txt
	rm input/temp.txt

	printf "\033[2m"
	for (( t=1; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		alpha=${alphas[t]}
		# if the simulation is for level 3
		if grep -Fxq "decoder 1,1,1" input/${ts}.txt;
		then
			# if the simulation is for level 3
			echo "REPLACE decoder 1,1,1 WITH decoder 4,4,4 IN input/${ts}.txt"
			replace "decoder 1,1,1" "decoder 4,4,4" input/${ts}.txt
		elif grep -Fxq "decoder 1,1" input/${ts}.txt;
		then
			# if the simulation is for level 2
			echo "REPLACE decoder 1,1 WITH decoder 4,4 IN input/${ts}.txt"
			replace "decoder 1,1" "decoder 4,4" input/${ts}.txt
		elif grep -Fxq "decoder 1" input/${ts}.txt;
		then
			# if the simulation is for level 1
			echo "REPLACE decoder 1 WITH decoder 4 IN input/${ts}.txt"
			replace "decoder 1" "decoder 4" input/${ts}.txt
		else
			:
		fi

		# set the alpha for dcfraction.
		echo "REPLACE dcfraction ${refalpha} WITH dcfraction ${alpha} IN input/${ts}.txt"
		replace "dcfraction ${refalpha}" "dcfraction ${alpha}" input/${ts}.txt

		echo "xxxxxxx"
	done
	printf "\033[0m"

elif [[ "$1" == "archive" ]]; then
	printf "\033[2m"
	echo "Reports in ${report_dir}"
	mkdir -p /Users/pavi/OneDrive\ -\ University\ of\ Waterloo/chbank/Nov4
	touch /Users/pavi/OneDrive\ -\ University\ of\ Waterloo/chbank/Nov4/${log}.txt
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		echo "Archiving ${ts}"
		ts=${timestamps[t]}
		cd ${outdir}
		# echo "Inside ${outdir}"
		tar -zcvf ${ts}.tar.gz ${ts}
		mv ${ts}.tar.gz /Users/pavi/OneDrive\ -\ University\ of\ Waterloo/chbank/Nov4/
		echo "${ts}" >> /Users/pavi/OneDrive\ -\ University\ of\ Waterloo/chbank/Nov4/${log}.txt
		cd ${chflowdir}
		cp input/${ts}.txt /Users/pavi/OneDrive\ -\ University\ of\ Waterloo/chbank/Nov4
		cp input/$schedule_${ts}.txt /Users/pavi/OneDrive\ -\ University\ of\ Waterloo/chbank/Nov4
	done
	printf "\033[0m"

elif [[ "$1" == "filter" ]]; then
	printf "\033[2m"
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		alpha=${alphas[t]}
		if [[ ! " ${untrust[@]} " =~ " ${alpha} " ]]; then
			echo ${ts}
		fi
	done
	printf "\033[0m"

elif [[ "$1" == "chmod" ]]; then
	printf "\033[2m"
	cd ${outdir}
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "Changing permissions for ${outdir}/${ts}"
		chmod -R 777 ${ts}
	done
	cd ${chflowdir}
	printf "\033[0m"

elif [[ "$1" == "zip" ]]; then
	cd ${outdir}
	rm -rf data/
	mkdir data/
	printf "\033[2m"
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "zipping ${ts}"
		tar -zcvf ${ts}.tar.gz ${ts}
		# move the folder and the input files into data
		mv ${ts}.tar.gz data/
		cp ${chflowdir}/input/${ts}.txt data/
		cp ${chflowdir}/input/schedule_${ts}.txt data/
	done
	# Put all the zipped folders in a zip folder.
	echo "Compressing data"
	tar -zcvf data.tar.gz data/
	cd ${chflowdir}
	printf "\033[0m"

elif [[ "$1" == "move_input" ]]; then
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "Copying from ${chflowdir}/input/${ts}.txt to /project/def-jemerson/input/"
		cp ${chflowdir}/input/${ts}.txt /project/def-jemerson/input/
		cp ${chflowdir}/input/schedule_${ts}.txt /project/def-jemerson/input/
	done

elif [[ "$1" == "gdrive" ]]; then
	printf "\033[2m"
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "Moving ${outdir}/${ts}.tar.gz to Google Drive"
		mv ${outdir}/${ts}.tar.gz /Users/pavi/Google\ Drive/channels_for_report/partial_decoders
		echo "Copying input/${ts}.txt to Google Drive"
		cp input/${ts}.txt /Users/pavi/Google\ Drive/channels_for_report/partial_decoders/
		echo "Copying input/schedule_${ts}.txt to Google Drive"
		cp input/schedule_${ts}.txt /Users/pavi/Google\ Drive/channels_for_report/partial_decoders/
	done
	printf "\033[0m"

elif [[ "$1" == "pmetrics" ]]; then
	# Compute physical infidelity for all channels.
	printf "\033[2m"
	touch input/temp.txt
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "sbload ${ts}" >> input/temp.txt
		echo "pmetrics infid" >> input/temp.txt
		echo "collect" >> input/temp.txt
	done
	echo "quit" >> input/temp.txt
	./chflow.sh -- temp.txt
	rm input/temp.txt
	printf "\033[0m"

elif [[ "$1" == "plot" ]]; then
	printf "\033[2m"
	echo "sbload ${refts}" > input/temp.txt
	printf -v joined_timestamps '%s,' "${timestamps[@]:1}"
	echo "nrplot 0 0 ${joined_timestamps%?}" >> input/temp.txt
	echo "dciplot infid infid ${joined_timestamps%?} 0" >> input/temp.txt
	echo "mcplot infid infid 0,1 0 ${joined_timestamps%?}" >> input/temp.txt
	echo "quit" >> input/temp.txt
	./chflow.sh -- temp.txt
	rm input/temp.txt
	printf "\033[0m"

elif [[ "$1" == "rename" ]]; then
	# Rename the channel directory to include the user and the cluster that generated it.
	if [ -n ${cluster} ]; then
		cd ${outdir}
		echo -e "\033[2mAdding ${local_user}_{cluster} to the channel folders for ${log}.\033[0m"
		for (( t=0; t<${#timestamps[@]}; ++t )); do
			ts=${timestamps[t]}
			mv ${ts}/ ${local_user}_${cluster}_${ts}/
		done
		cd ${chflowdir}
	else
		echo -e "\033[2mNo action.\033[0m"
	fi

elif [[ "$1" == "from_cluster" ]]; then
	# bring the data folder and unzip
	printf "\033[2m"
	echo "Bringing simulation from ${cluster}"
	scp -r ${local_user}@${cluster}.computecanada.ca:/project/def-jemerson/chbank/data.tar.gz ${outdir}
	cd ${outdir}
	tar -xvf data.tar.gz
	
	# unzip the individual datasets.
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "Trashing ${ts}"
		trash ${ts}
		cp data/${ts}.tar.gz .
		tar -xvf ${ts}.tar.gz
		trash ${ts}.tar.gz
		
		#### Copying input files to chflow
		echo "Copying input file ${ts}.txt from data"
		cp data/${ts}.txt ${chflowdir}/input/
		echo "Copying schedule_${ts}.txt from data"
		cp data/schedule_${ts}.txt ${chflowdir}/input/
		
		#### Prepare output directory after moving from cluster.
		echo "/project/def-jemerson/chbank WITH ${outdir} IN input/${ts}.txt"
		replace "\/project\/def-jemerson\/chbank" ${outdir//\//\\\/} ${chflowdir}/input/${ts}.txt
		
		echo "xxxxxxx"
	done
	printf "\033[0m"
	
	printf "\033[2m"
	# Add a new timestamp for the data record.
	datetime=$(date +%d_%m_%Y_%H_%M_%S)
	echo "Add the time stamp ${datetime} to the data folders so that it can be kept as a record."
	mv data "data_${datetime}"
	
	# Adding an information file for the folder.
	echo "Data folder: $(date)" > data_${datetime}/info.txt
	printf -v joined_timestamps '%s,' "${timestamps[@]:0}"
	echo "Timestamps: ${joined_timestamps%?}" >> data_${datetime}/info.txt
	printf -v joined_alphas '%s,' "${alphas[@]:0}"
	echo "Alphas: ${joined_alphas%?}" >> data_${datetime}/info.txt

	# Zipping data folder for records
	tar -zcvf "data_${datetime}.tar.gz" "data_${datetime}"
	
	printf "\033[0m"

	cd ${chflowdir}

else
	usage
fi
