#!/bin/bash
host=$(hostname)
echo "Host: $host"
cluster="$2"

if [[ $host == *"paviws"* ]]; then
	outdir="/Users/pavi/Documents/chbank"
	chflowdir="/Users/pavi/Documents/rclearn/chflow"
	# report_dir="/Users/pavi/OneDrive\ -\ University\ of\ Waterloo/chbank/Nov4"
	cores=$(sysctl -n hw.ncpu)
	sed_prepend="'' "
	ising_level3=("ising_l3_08_12_2020_00" "ising_l3_08_12_2020_01" "ising_l3_08_12_2020_02" "ising_l3_08_12_2020_03" "ising_l3_08_12_2020_04" "ising_l3_08_12_2020_05" "ising_l3_08_12_2020_06" "ising_l3_08_12_2020_07" "ising_l3_08_12_2020_08" "ising_l3_08_12_2020_09" "ising_l3_08_12_2020_10" "ising_l3_08_12_2020_11")
	npcorr_level2=("npcorr_l2_08_12_2020_00" "npcorr_l2_08_12_2020_01" "npcorr_l2_08_12_2020_02" "npcorr_l2_08_12_2020_03" "npcorr_l2_08_12_2020_04" "npcorr_l2_08_12_2020_05" "npcorr_l2_08_12_2020_06" "npcorr_l2_08_12_2020_07" "npcorr_l2_08_12_2020_08" "npcorr_l2_08_12_2020_09" "npcorr_l2_08_12_2020_10" "npcorr_l2_08_12_2020_11")
	pcorr_level2=("pcorr_l2_08_12_2020_00" "pcorr_l2_08_12_2020_01" "pcorr_l2_08_12_2020_02" "pcorr_l2_08_12_2020_03" "pcorr_l2_08_12_2020_04" "pcorr_l2_08_12_2020_05" "pcorr_l2_08_12_2020_06" "pcorr_l2_08_12_2020_07" "pcorr_l2_08_12_2020_08" "pcorr_l2_08_12_2020_09" "pcorr_l2_08_12_2020_10" "pcorr_l2_08_12_2020_11")
	pcorr_level3=("pcorr_l3_08_12_2020_00" "pcorr_l3_08_12_2020_01" "pcorr_l3_08_12_2020_02" "pcorr_l3_08_12_2020_03" "pcorr_l3_08_12_2020_04" "pcorr_l3_08_12_2020_05" "pcorr_l3_08_12_2020_06" "pcorr_l3_08_12_2020_07" "pcorr_l3_08_12_2020_08" "pcorr_l3_08_12_2020_09" "pcorr_l3_08_12_2020_10" "pcorr_l3_08_12_2020_11")
	cptp_level2=("cptp_l2_08_12_2020_00" "cptp_l2_08_12_2020_01" "cptp_l2_08_12_2020_02" "cptp_l2_08_12_2020_03" "cptp_l2_08_12_2020_04" "cptp_l2_08_12_2020_05" "cptp_l2_08_12_2020_06" "cptp_l2_08_12_2020_07" "cptp_l2_08_12_2020_08" "cptp_l2_08_12_2020_09" "cptp_l2_08_12_2020_10" "cptp_l2_08_12_2020_11")
	
	alphas=(0 0.0002 0.0004 0.0008 0.0016 0.0032 0.0063 0.0126 0.0251 0.0501 0.1 1)

elif [[ $host == "oem-ThinkPad-X1-Carbon-Gen-8" ]]; then
	outdir="/home/oem/Documents/chbank"
	chflowdir="/home/oem/Desktop/Research_PhD/chflow"
	cores=$(nproc --all)
	ising_level3=("ising_l3_08_12_2020_00" "ising_l3_08_12_2020_01" "ising_l3_08_12_2020_02" "ising_l3_08_12_2020_03" "ising_l3_08_12_2020_04" "ising_l3_08_12_2020_05" "ising_l3_08_12_2020_06" "ising_l3_08_12_2020_07")
	npcorr_level2=("npcorr_l2_08_12_2020_00" "npcorr_l2_08_12_2020_01" "npcorr_l2_08_12_2020_02" "npcorr_l2_08_12_2020_03" "npcorr_l2_08_12_2020_04" "npcorr_l2_08_12_2020_05" "npcorr_l2_08_12_2020_06" "npcorr_l2_08_12_2020_07")
	pcorr_level2=("pcorr_l2_08_12_2020_00" "pcorr_l2_08_12_2020_01" "pcorr_l2_08_12_2020_02" "pcorr_l2_08_12_2020_03" "pcorr_l2_08_12_2020_04" "pcorr_l2_08_12_2020_05" "pcorr_l2_08_12_2020_06" "pcorr_l2_08_12_2020_07")
	pcorr_level3=("pcorr_l3_08_12_2020_00" "pcorr_l3_08_12_2020_01" "pcorr_l3_08_12_2020_02" "pcorr_l3_08_12_2020_03" "pcorr_l3_08_12_2020_04" "pcorr_l3_08_12_2020_05" "pcorr_l3_08_12_2020_06" "pcorr_l3_08_12_2020_07")	
	cptp_level2=("cptp_l2_08_12_2020_00" "cptp_l2_08_12_2020_01" "cptp_l2_08_12_2020_02" "cptp_l2_08_12_2020_03" "cptp_l2_08_12_2020_04" "cptp_l2_08_12_2020_05" "cptp_l2_08_12_2020_06" "cptp_l2_08_12_2020_07")
	# alphas=(0 0.00013 0.00027 0.00093 0.00368 0.00391 0.00415 0.00678)
	alphas=(0 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1)

else
	outdir="/project/def-jemerson/chbank"
	chflowdir="/project/def-jemerson/${USER}/chflow"
	sed_prepend=""
fi

if [[ ! -z "$cluster" ]]; then
	cores=40 # 48 for cedar 40 for beluga and 32 for graham
	# Run this in Python to generate the time stamps
	# " ".join(["\"cptp_l2_08_12_2020_%.2d\"" % i for i in range(40)])
	ising_level3=("ising_l3_08_12_2020_00" "ising_l3_08_12_2020_01" "ising_l3_08_12_2020_02" "ising_l3_08_12_2020_03" "ising_l3_08_12_2020_04" "ising_l3_08_12_2020_05" "ising_l3_08_12_2020_06" "ising_l3_08_12_2020_07" "ising_l3_08_12_2020_08" "ising_l3_08_12_2020_09" "ising_l3_08_12_2020_10" "ising_l3_08_12_2020_11" "ising_l3_08_12_2020_12" "ising_l3_08_12_2020_13" "ising_l3_08_12_2020_14" "ising_l3_08_12_2020_15" "ising_l3_08_12_2020_16" "ising_l3_08_12_2020_17" "ising_l3_08_12_2020_18")
	npcorr_level2=("npcorr_l2_08_12_2020_00" "npcorr_l2_08_12_2020_01" "npcorr_l2_08_12_2020_02" "npcorr_l2_08_12_2020_03" "npcorr_l2_08_12_2020_04" "npcorr_l2_08_12_2020_05" "npcorr_l2_08_12_2020_06" "npcorr_l2_08_12_2020_07" "npcorr_l2_08_12_2020_08" "npcorr_l2_08_12_2020_09" "npcorr_l2_08_12_2020_10" "npcorr_l2_08_12_2020_11" "npcorr_l2_08_12_2020_12" "npcorr_l2_08_12_2020_13" "npcorr_l2_08_12_2020_14" "npcorr_l2_08_12_2020_15" "npcorr_l2_08_12_2020_16" "npcorr_l2_08_12_2020_17" "npcorr_l2_08_12_2020_18")
	pcorr_level2=("pcorr_l2_08_12_2020_00" "pcorr_l2_08_12_2020_01" "pcorr_l2_08_12_2020_02" "pcorr_l2_08_12_2020_03" "pcorr_l2_08_12_2020_04" "pcorr_l2_08_12_2020_05" "pcorr_l2_08_12_2020_06" "pcorr_l2_08_12_2020_07" "pcorr_l2_08_12_2020_08" "pcorr_l2_08_12_2020_09" "pcorr_l2_08_12_2020_10" "pcorr_l2_08_12_2020_11" "pcorr_l2_08_12_2020_12" "pcorr_l2_08_12_2020_13" "pcorr_l2_08_12_2020_14" "pcorr_l2_08_12_2020_15" "pcorr_l2_08_12_2020_16" "pcorr_l2_08_12_2020_17" "pcorr_l2_08_12_2020_18")
	pcorr_level3=("pcorr_l3_08_12_2020_00" "pcorr_l3_08_12_2020_01" "pcorr_l3_08_12_2020_02" "pcorr_l3_08_12_2020_03" "pcorr_l3_08_12_2020_04" "pcorr_l3_08_12_2020_05" "pcorr_l3_08_12_2020_06" "pcorr_l3_08_12_2020_07" "pcorr_l3_08_12_2020_08" "pcorr_l3_08_12_2020_09" "pcorr_l3_08_12_2020_10" "pcorr_l3_08_12_2020_11" "pcorr_l3_08_12_2020_12" "pcorr_l3_08_12_2020_13" "pcorr_l3_08_12_2020_14" "pcorr_l3_08_12_2020_15" "pcorr_l3_08_12_2020_16" "pcorr_l3_08_12_2020_17" "pcorr_l3_08_12_2020_18")
    cptp_level2=("cptp_l2_08_12_2020_00" "cptp_l2_08_12_2020_01" "cptp_l2_08_12_2020_02" "cptp_l2_08_12_2020_03" "cptp_l2_08_12_2020_04" "cptp_l2_08_12_2020_05" "cptp_l2_08_12_2020_06" "cptp_l2_08_12_2020_07" "cptp_l2_08_12_2020_08" "cptp_l2_08_12_2020_09" "cptp_l2_08_12_2020_10" "cptp_l2_08_12_2020_11" "cptp_l2_08_12_2020_12" "cptp_l2_08_12_2020_13" "cptp_l2_08_12_2020_14" "cptp_l2_08_12_2020_15" "cptp_l2_08_12_2020_16" "cptp_l2_08_12_2020_17" "cptp_l2_08_12_2020_18" "cptp_l2_08_12_2020_19" "cptp_l2_08_12_2020_20" "cptp_l2_08_12_2020_21" "cptp_l2_08_12_2020_22" "cptp_l2_08_12_2020_23" "cptp_l2_08_12_2020_24" "cptp_l2_08_12_2020_25" "cptp_l2_08_12_2020_26" "cptp_l2_08_12_2020_27" "cptp_l2_08_12_2020_28" "cptp_l2_08_12_2020_29" "cptp_l2_08_12_2020_30" "cptp_l2_08_12_2020_31" "cptp_l2_08_12_2020_32" "cptp_l2_08_12_2020_33" "cptp_l2_08_12_2020_34" "cptp_l2_08_12_2020_35" "cptp_l2_08_12_2020_36" "cptp_l2_08_12_2020_37" "cptp_l2_08_12_2020_38" "cptp_l2_08_12_2020_39")
    aditya_cptp_level2=("cptp_l2_24_12_2020_00" "cptp_l2_24_12_2020_01" "cptp_l2_24_12_2020_02" "cptp_l2_24_12_2020_03" "cptp_l2_24_12_2020_04" "cptp_l2_24_12_2020_06" "cptp_l2_24_12_2020_07" "cptp_l2_24_12_2020_08" "cptp_l2_24_12_2020_10" "cptp_l2_24_12_2020_11" "cptp_l2_24_12_2020_12" "cptp_l2_24_12_2020_13" "cptp_l2_24_12_2020_14" "cptp_l2_24_12_2020_15" "cptp_l2_24_12_2020_16" "cptp_l2_24_12_2020_17" "cptp_l2_24_12_2020_18" "cptp_l2_24_12_2020_19" "cptp_l2_24_12_2020_20" "cptp_l2_24_12_2020_21" "cptp_l2_24_12_2020_22" "cptp_l2_24_12_2020_23" "cptp_l2_24_12_2020_24" "cptp_l2_24_12_2020_25" "cptp_l2_24_12_2020_26" "cptp_l2_24_12_2020_27" "cptp_l2_24_12_2020_28" "cptp_l2_24_12_2020_29" "cptp_l2_24_12_2020_30" "cptp_l2_24_12_2020_31" "cptp_l2_24_12_2020_32" "cptp_l2_24_12_2020_33" "cptp_l2_24_12_2020_34" "cptp_l2_24_12_2020_35" "cptp_l2_24_12_2020_36" "cptp_l2_24_12_2020_37")
    # alphas=(0 0.001 0.0011 0.0013 0.0015 0.0016 0.0019 0.0021 0.0024 0.0027 0.0031 0.0035 0.0039 0.0045 0.005 0.0057 0.0065 0.0073 0.0083 0.0094 0.0106 0.0121 0.0137 0.0155 0.0175 0.0198 0.0225 0.0254 0.0288 0.0326 0.0369 0.0418 0.0474 0.0537 0.0608 0.0688 0.078 0.0883 0.1 1)
    alphas=(0 0.001 0.00115396  0.00133162  0.00153663  0.00236124  0.00272477  0.00314427  0.00483159  0.00557545  0.00643384  0.00742438 0.00856742  0.00988644  0.01140854  0.01316497  0.01519183 0.01753073  0.02022973  0.02334425  0.02693829  0.03108565 0.03587154  0.04139425  0.04776722  0.05512137  0.06360774 0.07340067  0.08470128  0.09774172  0.11278984  0.13015474 0.15019311  0.17331653  0.2 1)
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

display() {
	./chflow.sh $ts
}

timestamps=("${aditya_cptp_level2[@]}")
log=aditya_cptp_level2
refts=${timestamps[0]}

if [[ "$1" == "overwrite" ]]; then
	rm input/partial_decoders_${log}.txt
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		rerun $ts $log
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
		sbcmds=("#!/bin/bash" "#SBATCH --account=def-jemerson" "#SBATCH --begin=now" "#SBATCH --nodes=1" "#SBATCH --time=05:00:00" "#SBATCH --ntasks-per-node=48" "#SBATCH -o /project/def-jemerson/chbank/${USER}_${log}_output.o" "#SBATCH -e /project/def-jemerson/chbank/${USER}_${log}_errors.o" "#SBATCH --mail-type=ALL" "#SBATCH --mail-user=pavithran.sridhar@gmail.com")
		for (( s=0; s<${#sbcmds[@]}; ++s )); do
			echo "${sbcmds[s]}" >> input/${cluster}/partial_decoders_${log}.sh
		done
		echo "module load intel python scipy-stack" >> input/${cluster}/partial_decoders_${log}.sh
		echo "cd /project/def-jemerson/${USER}/chflow" >> input/${cluster}/partial_decoders_${log}.sh
		echo "parallel --joblog partial_decoders_${log}.log ./chflow.sh {1} :::: input/partial_decoders_${log}.txt" >> input/${cluster}/partial_decoders_${log}.sh
		echo "xxxxxxx"
		echo "Run the following command."
		echo "sbatch input/${cluster}/partial_decoders_${log}.sh"
	fi

elif [[ "$1" == "schedule_copy" ]]; then
	for (( t=1; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo -e "\033[2mremoving ./input/schedule_${ts}.txt\033[0m"
		rm ./input/schedule_${ts}.txt
		echo -e "\033[2mcopying ./input/schedule_${timestamps[0]}.txt ./input/schedule_${ts}.txt\033[0m"
		cp ./input/schedule_${timestamps[0]}.txt ./input/schedule_${ts}.txt
		echo "xxxxxxx"
	done

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

		echo "REPLACE decoder 1,1 WITH decoder 4,4 IN input/${ts}.txt"
		sed -i ${sed_prepend}"s/decoder 1,1/decoder 4,4/g" input/${ts}.txt
		echo "REPLACE dcfraction ${refalpha} WITH dcfraction ${alpha} IN input/${ts}.txt"
		sed -i ${sed_prepend}"s/dcfraction ${refalpha}/dcfraction ${alpha}/g" input/${ts}.txt

		# echo "REPLACE decoder 1,1,1 WITH decoder 3,3,3 IN input/${ts}.txt"
		# sed -i ${sed_prepend}"s/decoder 1,1,1/decoder 3,3,3/g" input/${ts}.txt
		# echo "REPLACE dcfraction ${refalpha} WITH dcfraction ${alpha} IN input/${ts}.txt"
		# sed -i ${sed_prepend}"s/dcfraction ${refalpha}/dcfraction ${alpha}/g" input/${ts}.txt

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
	echo "dciplot infid infid ${joined_timestamps%?} 0" >> input/temp.txt
	echo "mcplot infid infid 0,1 0 ${joined_timestamps%?}" >> input/temp.txt
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
		scp /Users/pavi/Documents/chbank/${ts}.tar.gz pavi@${cluster}.computecanada.ca:/project/def-jemerson/chbank/
		echo "Sending input file ${ts}.txt to cluster"
		scp /Users/pavi/Documents/rclearn/chflow/input/$ts.txt pavi@${cluster}.computecanada.ca:/project/def-jemerson/pavi/chflow/input/
		echo "Sending input file schedule_${ts}.txt to cluster"
		scp /Users/pavi/Documents/rclearn/chflow/input/schedule_$ts.txt pavi@${cluster}.computecanada.ca:/project/def-jemerson/pavi/chflow/input/
	done

elif [[ "$1" == "from_cluster" ]]; then
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "Trashing ${ts}"
		trash ${outdir}/$ts
		echo "Bringing output ${ts} from cluster"
		scp -r pavi@${cluster}.computecanada.ca:/project/def-jemerson/chbank/$ts.tar.gz ${outdir}
		cd ${outdir}
		tar -xvf $ts.tar.gz
		cd ${chflowdir}
		#### Bring from chflow
		echo "Bringing input file ${ts}.txt from cluster"
		scp pavi@${cluster}.computecanada.ca:/project/def-jemerson/pavi/chflow/input/$ts.txt ${chflowdir}/input
		echo "Bringing schedule_${ts}.txt from cluster"
		scp pavi@${cluster}.computecanada.ca:/project/def-jemerson/pavi/chflow/input/schedule_$ts.txt ${chflowdir}/input
		#### Bring from def-jemerson/input
		# echo "Bringing input file ${ts}.txt from cluster"
		# scp pavi@${cluster}.computecanada.ca:/project/def-jemerson/input/$ts.txt ${chflowdir}/input
		# echo "Bringing schedule_${ts}.txt from cluster"
		# scp pavi@${cluster}.computecanada.ca:/project/def-jemerson/input/schedule_$ts.txt ${chflowdir}/input
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
