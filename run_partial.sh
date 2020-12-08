#!/bin/bash
host=$(hostname)
echo "Host: $host"

if [[ $host == *"paviws"* ]]; then
	outdir="/Users/pavi/Documents/chbank"
	chflowdir="/Users/pavi/Dropbox/rclearn/chflow"
	# report_dir="/Users/pavi/OneDrive\ -\ University\ of\ Waterloo/chbank/Nov4"
	cores=$(sysctl -n hw.ncpu)
	sed_prepend="'' "
	ising_level3=("ising_l3_08_12_2020_00" "ising_l3_08_12_2020_01" "ising_l3_08_12_2020_02" "ising_l3_08_12_2020_03" "ising_l3_08_12_2020_04" "ising_l3_08_12_2020_05" "ising_l3_08_12_2020_06" "ising_l3_08_12_2020_07" "ising_l3_08_12_2020_08" "ising_l3_08_12_2020_09" "ising_l3_08_12_2020_10" "ising_l3_08_12_2020_11")
	npcorr_level2=("npcorr_l2_08_12_2020_00" "npcorr_l2_08_12_2020_01" "npcorr_l2_08_12_2020_02" "npcorr_l2_08_12_2020_03" "npcorr_l2_08_12_2020_04" "npcorr_l2_08_12_2020_05" "npcorr_l2_08_12_2020_06" "npcorr_l2_08_12_2020_07" "npcorr_l2_08_12_2020_08" "npcorr_l2_08_12_2020_09" "npcorr_l2_08_12_2020_10" "npcorr_l2_08_12_2020_11")
	pcorr_level2=("pcorr_l2_08_12_2020_00" "pcorr_l2_08_12_2020_01" "pcorr_l2_08_12_2020_02" "pcorr_l2_08_12_2020_03" "pcorr_l2_08_12_2020_04" "pcorr_l2_08_12_2020_05" "pcorr_l2_08_12_2020_06" "pcorr_l2_08_12_2020_07" "pcorr_l2_08_12_2020_08" "pcorr_l2_08_12_2020_09" "pcorr_l2_08_12_2020_10" "pcorr_l2_08_12_2020_11")
	pcorr_level3=("pcorr_l3_08_12_2020_00" "pcorr_l3_08_12_2020_01" "pcorr_l3_08_12_2020_02" "pcorr_l3_08_12_2020_03" "pcorr_l3_08_12_2020_04" "pcorr_l3_08_12_2020_05" "pcorr_l3_08_12_2020_06" "pcorr_l3_08_12_2020_07" "pcorr_l3_08_12_2020_08" "pcorr_l3_08_12_2020_09" "pcorr_l3_08_12_2020_10" "pcorr_l3_08_12_2020_11")
	# ("29_10_2020_13_17_31" "29_10_2020_13_17_32" "29_10_2020_13_17_34" "29_10_2020_13_17_36" "29_10_2020_13_17_38" "29_10_2020_13_17_40" "29_10_2020_13_17_42" "29_10_2020_13_17_44" "29_10_2020_13_17_50" "29_10_2020_13_17_53" "29_10_2020_13_17_63" "29_10_2020_13_17_66")
    ## Cluster runs
    alphas=(0 0.00021 0.00027 0.00135 0.00326 0.00378 0.00391 0.00415 0.00427 0.00467 0.00678 1)

elif [[ $host == "oem-ThinkPad-X1-Carbon-Gen-8" ]]; then
	outdir="/home/oem/Documents/chbank"
	chflowdir="/home/oem/Desktop/Research_PhD/chflow"
	cores=$(nproc --all)
	ising_level3=("ising_l3_08_12_2020_00" "ising_l3_08_12_2020_01" "ising_l3_08_12_2020_02" "ising_l3_08_12_2020_03" "ising_l3_08_12_2020_04" "ising_l3_08_12_2020_05" "ising_l3_08_12_2020_06" "ising_l3_08_12_2020_07")
	npcorr_level2=("npcorr_l2_08_12_2020_00" "npcorr_l2_08_12_2020_01" "npcorr_l2_08_12_2020_02" "npcorr_l2_08_12_2020_03" "npcorr_l2_08_12_2020_04" "npcorr_l2_08_12_2020_05" "npcorr_l2_08_12_2020_06" "npcorr_l2_08_12_2020_07")
	pcorr_level2=("pcorr_l2_08_12_2020_00" "pcorr_l2_08_12_2020_01" "pcorr_l2_08_12_2020_02" "pcorr_l2_08_12_2020_03" "pcorr_l2_08_12_2020_04" "pcorr_l2_08_12_2020_05" "pcorr_l2_08_12_2020_06" "pcorr_l2_08_12_2020_07")
	pcorr_level3=("pcorr_l3_08_12_2020_00" "pcorr_l3_08_12_2020_01" "pcorr_l3_08_12_2020_02" "pcorr_l3_08_12_2020_03" "pcorr_l3_08_12_2020_04" "pcorr_l3_08_12_2020_05" "pcorr_l3_08_12_2020_06" "pcorr_l3_08_12_2020_07")
	alphas=(0 0.00013 0.00027 0.00093 0.00368 0.00391 0.00415 0.00678)
else
	outdir="/project/def-jemerson/chbank"
	chflowdir="/project/def-jemerson/${USER}/chflow"
	cores=48
	ising_level3_imp_final_timestamps=("29_10_2020_13_17_31" "29_10_2020_13_17_32" "29_10_2020_13_17_34" "29_10_2020_13_17_36" "29_10_2020_13_17_38" "29_10_2020_13_17_40" "29_10_2020_13_17_42" "29_10_2020_13_17_44" "29_10_2020_13_17_50" "29_10_2020_13_17_53" "29_10_2020_13_17_63" "29_10_2020_13_17_66" "29_10_2020_13_17_78" "29_10_2020_13_17_79" "29_10_2020_13_17_80" "29_10_2020_13_17_81" "29_10_2020_13_17_82" "29_10_2020_13_17_83" "31_10_2020_13_17_54")
    alphas=(0 0.0001 0.00013 0.00021 0.00027 0.00035 0.00044 0.00093 0.00135 0.00326 0.00368 0.00378 0.00391 0.00403 0.00415 0.00427 0.00467 0.00678 1)
	sed_prepend=""
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
	echo "Running $1"
	echo "${ts}" >> input/partial_decoders.txt
}

display() {
	./chflow.sh $ts
}

timestamps=("${ising_level3_imp_final_timestamps[@]}")
log=ising_level3_imp_final_timestamps
refts=${timestamps[0]}

if [[ "$1" == "overwrite" ]]; then
	rm input/partial_decoders.txt
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		rerun $ts
		echo "xxxxxxx"
	done
	if [[ $host == *"paviws"* ]]; then
		echo "parallel --joblog partial_decoders.log --jobs ${cores} ./chflow.sh {1} :::: input/partial_decoders.txt"
	elif [[ $host == "oem-ThinkPad-X1-Carbon-Gen-8" ]]; then
		echo "parallel --joblog partial_decoders.log --jobs ${cores} ./chflow.sh {1} :::: input/partial_decoders.txt"
	else
		rm input/cedar/partial_decoders.sh
		sbcmds=("#!/bin/bash" "#SBATCH --account=def-jemerson" "#SBATCH --begin=now" "#SBATCH --nodes=1" "#SBATCH --time=05:00:00" "#SBATCH --ntasks-per-node=48" "#SBATCH -o /project/def-jemerson/chbank/${USER}_partial_output.o" "#SBATCH -e /project/def-jemerson/chbank/${USER}_partial_errors.o" "#SBATCH --mail-type=ALL" "#SBATCH --mail-user=pavithran.sridhar@gmail.com")
		for (( s=0; s<${#sbcmds[@]}; ++s )); do
			echo "${sbcmds[s]}" >> input/cedar/partial_decoders.sh
		done
		echo "module load intel python scipy-stack" >> input/cedar/partial_decoders.sh
		echo "cd /project/def-jemerson/pavi/chflow" >> input/cedar/partial_decoders.sh
		echo "parallel --joblog partial_decoders.log ./chflow.sh {1} :::: input/partial_decoders.txt" >> input/cedar/partial_decoders.sh
		echo "sbatch input/cedar/partial_decoders.sh"
	fi
elif [[ "$1" == "generate" ]]; then
	refalpha=${alphas[0]}
	echo "sbload ${refts}" > input/temp.txt
	for (( t=1; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		alpha=${alphas[t]}
		echo "alpha = ${alpha}"
		echo -e "\033[2mremoving ${outdir}/${ts}/physical/*\033[0m"
		rm ${outdir}/${ts}/physical/*
		# echo "sbtwirl" >> input/temp.txt
		echo "submit ${ts}" >> input/temp.txt
	done

	echo "quit" >> input/temp.txt
	cat input/temp.txt
	./chflow.sh -- temp.txt
	rm input/temp.txt

	for (( t=1; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		alpha=${alphas[t]}
		# echo "REPLACE decoder 1 WITH decoder 3 IN input/${ts}.txt"
		# sed -i ${sed_prepend}"s/decoder 1/decoder 3/g" input/${ts}.txt
		# echo "REPLACE dcfraction ${refalpha} WITH dcfraction ${alpha} IN input/${ts}.txt"
		# sed -i ${sed_prepend}"s/dcfraction ${refalpha}/dcfraction ${alpha}/g" input/${ts}.txt

		#echo "REPLACE decoder 1,1 WITH decoder 3,3 IN input/${ts}.txt"
		#sed -i ${sed_prepend}"s/decoder 1,1/decoder 3,3/g" input/${ts}.txt
		#echo "REPLACE dcfraction ${refalpha} WITH dcfraction ${alpha} IN input/${ts}.txt"
		#sed -i ${sed_prepend}"s/dcfraction ${refalpha}/dcfraction ${alpha}/g" input/${ts}.txt

		echo "REPLACE decoder 1,1,1 WITH decoder 3,3,3 IN input/${ts}.txt"
		sed -i ${sed_prepend}"s/decoder 1,1,1/decoder 3,3,3/g" input/${ts}.txt
		echo "REPLACE dcfraction ${refalpha} WITH dcfraction ${alpha} IN input/${ts}.txt"
		sed -i ${sed_prepend}"s/dcfraction ${refalpha}/dcfraction ${alpha}/g" input/${ts}.txt

		# echo "REPLACE ecc Steane WITH ecc Steane,Steane IN input/${ts}.txt"
		# sed -i ${sed_prepend}"s/ecc Steane/ecc Steane,Steane/g" input/${ts}.txt

		echo "xxxxxxx"
	done
elif [[ "$1" == "archive" ]]; then
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
elif [[ "$1" == "filter" ]]; then
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		alpha=${alphas[t]}
		if [[ ! " ${untrust[@]} " =~ " ${alpha} " ]]; then
			echo ${ts}
		fi
	done
elif [[ "$1" == "chmod" ]]; then
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "Changing permissions for ${outdir}/${ts}"
		cd ${outdir}
		chmod -R 777 ${ts}
		cd ${chflowdir}
	done
elif [[ "$1" == "zip" ]]; then
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "zipping ${ts}"
		cd ${outdir}
		tar -zcvf ${ts}.tar.gz ${ts}
		cd ${chflowdir}
	done
elif [[ "$1" == "move_input" ]]; then
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "Copying from ${chflowdir}/input/${ts}.txt to /project/def-jemerson/input/"
		cp ${chflowdir}/input/${ts}.txt /project/def-jemerson/input/
		cp ${chflowdir}/input/schedule_${ts}.txt /project/def-jemerson/input/
	done
elif [[ "$1" == "gdrive" ]]; then
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "Moving ${outdir}/${ts}.tar.gz to Google Drive"
		mv ${outdir}/${ts}.tar.gz /Users/pavi/Google\ Drive/channels_for_report/partial_decoders
		echo "Copying input/${ts}.txt to Google Drive"
		cp input/${ts}.txt /Users/pavi/Google\ Drive/channels_for_report/partial_decoders/
		echo "Copying input/schedule_${ts}.txt to Google Drive"
		cp input/schedule_${ts}.txt /Users/pavi/Google\ Drive/channels_for_report/partial_decoders/
	done
elif [[ "$1" == "unzip_local" ]]; then
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "unzipping /Users/pavi/Documents/chbank/${ts}.tar.gz"
		tar -xvf /Users/pavi/Documents/chbank/${ts}.tar.gz
	done
elif [[ "$1" == "unzip_cluster" ]]; then
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "unzipping /project/def-jemerson/chbank/${ts}.tar.gz"
		tar -xvf /project/def-jemerson/chbank/${ts}.tar.gz
	done
elif [[ "$1" == "pmetrics" ]]; then
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "sbload ${ts}" > input/temp.txt
		echo "pmetrics infid" >> input/temp.txt
		echo "collect" >> input/temp.txt
		echo "quit" >> input/temp.txt
		./chflow.sh -- temp.txt
		rm input/temp.txt
	done
elif [[ "$1" == "plot" ]]; then
	echo "sbload ${refts}" > input/temp.txt
	printf -v joined_timestamps '%s,' "${timestamps[@]:1}"
	echo "dciplot infid infid ${joined_timestamps%?}" >> input/temp.txt
	echo "mcplot infid infid 1 1 ${joined_timestamps%?}" >> input/temp.txt
	echo "quit" >> input/temp.txt
	./chflow.sh -- temp.txt
	rm input/temp.txt
elif [[ "$1" == "to_cluster" ]]; then
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "/Users/pavi/Documents WITH /project/def-jemerson IN input/${ts}.txt"
		sed -i ${sed_prepend}"s/\/Users\/pavi\/Documents/\/project\/def-jemerson/g" input/${ts}.txt
		echo "zipping ${outdir}/${ts}"
		tar -zcvf ${outdir}/${ts}.tar.gz ${outdir}/${ts}
		echo "Sending output ${ts} to cluster"
		scp /Users/pavi/Documents/chbank/${ts}.tar.gz pavi@cedar.computecanada.ca:/project/def-jemerson/chbank/
		echo "Sending input file ${ts}.txt to cluster"
		scp /Users/pavi/Dropbox/rclearn/chflow/input/$ts.txt pavi@cedar.computecanada.ca:/project/def-jemerson/pavi/chflow/input/
		echo "Sending input file schedule_${ts}.txt to cluster"
		scp /Users/pavi/Dropbox/rclearn/chflow/input/schedule_$ts.txt pavi@cedar.computecanada.ca:/project/def-jemerson/pavi/chflow/input/
	done
elif [[ "$1" == "from_cluster" ]]; then
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "Trashing ${ts}"
		trash ${outdir}/$ts
		echo "Bringing output ${ts} from cluster"
		scp -r pavi@cedar.computecanada.ca:/project/def-jemerson/chbank/$ts.tar.gz ${outdir}
		cd ${outdir}
		tar -xvf $ts.tar.gz
		cd ${chflowdir}
		#### Bring from chflow
		echo "Bringing input file ${ts}.txt from cluster"
		scp pavi@cedar.computecanada.ca:/project/def-jemerson/pavi/chflow/input/$ts.txt ${chflowdir}/input
		echo "Bringing schedule_${ts}.txt from cluster"
		scp pavi@cedar.computecanada.ca:/project/def-jemerson/pavi/chflow/input/schedule_$ts.txt ${chflowdir}/input
		#### Bring from def-jemerson/input
		# echo "Bringing input file ${ts}.txt from cluster"
		# scp pavi@cedar.computecanada.ca:/project/def-jemerson/input/$ts.txt ${chflowdir}/input
		# echo "Bringing schedule_${ts}.txt from cluster"
		# scp pavi@cedar.computecanada.ca:/project/def-jemerson/input/schedule_$ts.txt ${chflowdir}/input
		#### Prepare output directory after moving from cluster.
		echo "/project/def-jemerson WITH /Users/pavi/Documents IN input/${ts}.txt"
		sed -i ${sed_prepend}"s/\/project\/def-jemerson/\/Users\/pavi\/Documents/g" input/${ts}.txt
		#### Prepare output directory after moving from cluster with different path.
		# echo "/home/a77jain/projects/def-jemerson WITH /Users/pavi/Documents IN input/${ts}.txt"
		# sed -i ${sed_prepend}"s/\/home\/a77jain\/projects\/def-jemerson/\/Users\/pavi\/Documents/g" input/${ts}.txt
	done
elif [[ "$1" == "display" ]]; then
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		display $ts
		echo "xxxxxxx"
	done
else
	echo "Unknown option."
fi
