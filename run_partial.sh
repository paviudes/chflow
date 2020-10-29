#!/bin/bash
host=$(hostname)
echo "Host: $host"
if [[ $host == *"paviws"* ]]; then
	outdir="/Users/pavi/Documents"
	chflowdir="/Users/pavi/Dropbox/rclearn"
	cores=$(sysctl -n hw.ncpu)
	sed_prepend="'' "
	nonpauli_timestamps=("21_07_2020_16_28_27" "20_07_2020_20_11_41" "21_07_2020_21_49_52" "03_08_2020_22_47_51" "03_08_2020_23_03_35" "03_08_2020_22_54_43" "04_08_2020_18_31_52" "04_08_2020_18_49_45" "04_08_2020_19_00_34" "21_07_2020_16_28_30" "25_08_2020_01_24_08" "25_08_2020_01_24_23")
	pauli_timestamps=("10_08_2020_00_33_27" "10_08_2020_00_33_34" "10_08_2020_00_33_35" "10_08_2020_00_33_36" "10_08_2020_10_04_35" "10_08_2020_10_04_37" "10_08_2020_16_53_00" "10_08_2020_16_53_01" "10_08_2020_16_53_02" "10_08_2020_16_53_03" "10_08_2020_16_53_04" "10_08_2020_16_53_05")
	alphas=(0 0.0001 0.0003 0.0005 0.0007 0.0012 0.0025 0.0036 0.005 0.01 0.02 1)
	# ext_pauli_timestamps=("10_08_2020_00_33_27" "15_09_2020_12_50_00" "15_09_2020_12_50_01" "15_09_2020_12_50_02" "15_09_2020_12_50_03" "15_09_2020_12_50_04" "15_09_2020_12_50_05" "15_09_2020_12_50_06" "15_09_2020_12_50_07" "15_09_2020_12_50_08")
	# alphas=(0 0.0013 0.0015 0.0017 0.0019 0.0021 0.0023 0.0026 0.0029 0.0032)
elif [[ $host == "oem-ThinkPad-X1-Carbon-Gen-8" ]]; then
	outdir="/home/oem/Documents"
	chflowdir="/home/oem/Desktop/Research_PhD"
	cores=$(nproc --all)
	nonpauli_timestamps=("21_07_2020_16_28_27" "03_08_2020_22_54_43" "04_08_2020_18_31_52" "04_08_2020_18_49_45" "04_08_2020_19_00_34" "21_07_2020_16_28_30" "25_08_2020_01_24_08" "25_08_2020_01_24_23")
	pauli_timestamps=("10_08_2020_00_33_27" "10_08_2020_10_04_37" "10_08_2020_16_53_00" "10_08_2020_16_53_01" "10_08_2020_16_53_02" "10_08_2020_16_53_03" "10_08_2020_16_53_04" "10_08_2020_16_53_05")
	alphas=(0 0.0001 0.0003 0.0007 0.0012 0.005 0.02 1)
else
	outdir="/project/def-jemerson/chbank"
	chflowdir="/project/def-jemerson/pavi/chflow"
	nonpauli_timestamps=("21_07_2020_16_28_27" "20_07_2020_20_11_41" "21_07_2020_21_49_52" "03_08_2020_22_47_51" "03_08_2020_23_03_35" "03_08_2020_22_54_43" "04_08_2020_18_31_52" "04_08_2020_18_49_45" "04_08_2020_19_00_34" "21_07_2020_16_28_30" "25_08_2020_01_24_08" "25_08_2020_01_24_23")
	# cores=$(nproc --all)
	alphas=(0 0.0001 0.0003 0.0005 0.0007 0.0012 0.0025 0.0036 0.005 0.01 0.02 1)
	sed_prepend=""
fi


rerun() {
	echo "removing ${outdir}/chbank/$1/channels/*"
	rm ${outdir}/chbank/$1/channels/*
	rm ${outdir}/chbank/$1/metrics/*
	rm ${outdir}/chbank/$1/results/*.npy
	# rm ${outdir}/chbank/$1/physical/*.npy
	echo "Running $1"
	echo "${ts}" >> input/partial_decoders.txt
}

display() {
	./chflow.sh $ts
}

timestamps=("${nonpauli_timestamps[@]}")

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
		sbcmds=("#!/bin/bash" "#SBATCH --account=def-jemerson" "#SBATCH --begin=now" "#SBATCH --nodes=1" "#SBATCH --time=05:00:00" "#SBATCH --ntasks-per-node=48" "#SBATCH -o /project/def-jemerson/chbank/partial_output.o" "#SBATCH -e /project/def-jemerson/chbank/partial_errors.o" "#SBATCH --mail-type=ALL" "#SBATCH --mail-user=pavithran.sridhar@gmail.com")
		for (( s=0; s<${#sbcmds[@]}; ++s )); do
			echo "${sbcmds[s]}" >> input/cedar/partial_decoders.sh
		done
		echo "module load intel python scipy-stack" >> input/cedar/partial_decoders.sh
		echo "cd /project/def-jemerson/pavi/chflow" >> input/cedar/partial_decoders.sh
		echo "parallel --joblog partial_decoders.log ./chflow.sh {1} :::: input/partial_decoders.txt" >> input/cedar/partial_decoders.sh
		echo "sbatch input/cedar/partial_decoders.sh"
	fi
elif [[ "$1" == "generate" ]]; then
	refts=${timestamps[0]}
	refalpha=${alphas[0]}
	for (( t=1; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		alpha=${alphas[t]}
		echo "alpha = ${alpha}"

		echo -e "\033[2mremoving ${outdir}/chbank/${ts}/physical/*\033[0m"
		rm ${outdir}/chbank/${ts}/physical/*

		echo "sbload ${refts}" > input/temp.txt
		# echo "sbtwirl" >> input/temp.txt
		echo "submit ${ts}" >> input/temp.txt
		echo "quit" >> input/temp.txt
		cat input/temp.txt
		./chflow.sh -- temp.txt
		rm input/temp.txt

		# echo "REPLACE decoder 1 WITH decoder 3 IN input/${ts}.txt"
		# sed -i ${sed_prepend}"s/decoder 1/decoder 3/g" input/${ts}.txt
		# echo "REPLACE dcfraction ${refalpha} WITH dcfraction ${alpha} IN input/${ts}.txt"
		# sed -i ${sed_prepend}"s/dcfraction ${refalpha}/dcfraction ${alpha}/g" input/${ts}.txt

		echo "REPLACE decoder 1,1 WITH decoder 3,3 IN input/${ts}.txt"
		sed -i ${sed_prepend}"s/decoder 1,1/decoder 3,3/g" input/${ts}.txt
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
		chmod -R 777 ${outdir}/${ts}
	done
elif [[ "$1" == "zip" ]]; then
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "zipping ${outdir}/chbank/${ts}"
		tar -zcvf ${outdir}/chbank/${ts}.tar.gz ${outdir}/chbank/${ts}
	done
elif [[ "$1" == "gdrive" ]]; then
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "Moving ${outdir}/chbank/${ts}.tar.gz to Google Drive"
		mv ${outdir}/chbank/${ts}.tar.gz /Users/pavi/Google\ Drive/channels_for_report/partial_decoders
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
elif [[ "$1" == "to_cluster" ]]; then
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "/Users/pavi/Documents WITH /project/def-jemerson IN input/${ts}.txt"
		sed -i ${sed_prepend}"s/\/Users\/pavi\/Documents/\/project\/def-jemerson/g" input/${ts}.txt
		echo "zipping ${outdir}/${ts}"
		tar -zcvf ${outdir}/chbank/${ts}.tar.gz ${outdir}/chbank/${ts}
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
		trash /Users/pavi/Documents/chbank/$ts
		echo "Bringing output ${ts} from cluster"
		scp -r pavi@cedar.computecanada.ca:/project/def-jemerson/chbank/$ts.tar.gz /Users/pavi/Documents/chbank
		tar -xvf /Users/pavi/Documents/chbank/$ts.tar.gz
		echo "Bringing input file ${ts}.txt from cluster"
		scp pavi@cedar.computecanada.ca:/project/def-jemerson/pavi/chflow/input/$ts.txt /Users/pavi/Dropbox/rclearn/chflow/input
		echo "Bringing schedule_${ts}.txt from cluster"
		scp pavi@cedar.computecanada.ca:/project/def-jemerson/pavi/chflow/input/schedule_$ts.txt /Users/pavi/Dropbox/rclearn/chflow/input
		echo "/project/def-jemerson WITH /Users/pavi/Documents IN input/${ts}.txt"
		sed -i ${sed_prepend}"s/\/project\/def-jemerson/\/Users\/pavi\/Documents/g" input/${ts}.txt
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
