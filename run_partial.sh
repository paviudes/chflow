#!/bin/bash
host=$(hostname)
echo "Host: $host"
if [[ $host == *"paviws"* ]]; then
	outdir="/Users/pavi/Documents"
	chflowdir="/Users/pavi/Dropbox/rclearn"
	cores=$(sysctl -n hw.ncpu)
else
	outdir="/project/def-jemerson/chbank"
	chflowdir="/project/def-jemerson/pavi/chflow"
	cores=$(nproc --all)
fi

rerun() {
	echo "removing ${outdir}/$1/channels/*"
	rm ${outdir}/$1/channels/*
	rm ${outdir}/$1/metrics/*
	rm ${outdir}/$1/results/*.npy
	echo "Running $1"
	echo "${ts}" >> input/partial_decoders.txt
}

display() {
	./chflow.sh $ts
}

pauli_timestamps=("21_07_2020_16_28_27" "20_07_2020_20_11_41" "21_07_2020_21_49_52" "03_08_2020_22_47_51" "03_08_2020_23_03_35" "03_08_2020_22_54_43" "04_08_2020_18_31_52" "04_08_2020_18_49_45" "04_08_2020_19_00_34" "21_07_2020_16_28_30")

cptp_timestamps=("06_08_2020_19_31_31" "06_08_2020_19_31_32" "06_08_2020_19_31_34" "06_08_2020_19_31_35" "06_08_2020_19_31_37" "06_08_2020_19_31_49" "06_08_2020_19_31_50" "06_08_2020_19_31_52" "06_08_2020_19_31_53" "06_08_2020_19_31_55")

rtasu_timestamps=("07_08_2020_00_44_26" "07_08_2020_00_44_27" "07_08_2020_00_44_29" "07_08_2020_00_44_30" "07_08_2020_00_44_31" "07_08_2020_00_44_33" "07_08_2020_00_44_36" "07_08_2020_00_44_38" "07_08_2020_00_44_39" "07_08_2020_00_44_40")

nonpauli_timestamps=("10_08_2020_00_33_27" "10_08_2020_00_33_29" "10_08_2020_00_33_30" "10_08_2020_00_33_31" "10_08_2020_00_33_33" "10_08_2020_00_33_34" "10_08_2020_00_33_35" "10_08_2020_00_33_36" "10_08_2020_10_04_35" "10_08_2020_10_04_37" "10_08_2020_16_53_00" "10_08_2020_16_53_01" "10_08_2020_16_53_02" "10_08_2020_16_53_03" "10_08_2020_16_53_04" "10_08_2020_16_53_05" "10_08_2020_16_53_06" "10_08_2020_16_53_07" "10_08_2020_16_53_08" "10_08_2020_10_04_38" "10_08_2020_10_04_39")

timestamps=("${nonpauli_timestamps[@]}")

alphas=(0 0.0001 0.0002 0.0003 0.0004 0.0007 0.0012 0.0019 0.0031 0.005 0.0085 0.0144 0.0245 0.0416 0.0707 0.1201 0.204 0.3466 0.5887 1 "ML")

if [[ "$1" == "overwrite" ]]; then
	rm input/partial_decoders.txt
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		rerun $ts
		echo "xxxxxxx"
	done
	echo "parallel --joblog partial_decoders.log --jobs ${cores} ./chflow.sh {1} :::: input/partial_decoders.txt"
elif [[ "$1" == "generate" ]]; then
	refts=${timestamps[0]}
	refalpha=${alphas[0]}
	for (( t=1; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		alpha=${alphas[t]}
		echo "alpha = ${alpha}"
		
		echo -e "\033[2mremoving ${outdir}/${ts}/physical/*\033[0m"
		rm ${outdir}/${ts}/physical/*

		echo "sbload ${refts}" > input/temp.txt
		echo "submit ${ts}" >> input/temp.txt
		echo "quit" >> input/temp.txt
		cat input/temp.txt
		./chflow.sh -- temp.txt
		rm input/temp.txt

		if [[ "$alpha" == "ML" ]]; then
			echo "REPLACE decoder 1,1 WITH decoder 0,0 IN input/${ts}.txt"
			sed -i '' "s/decoder 1,1/decoder 0,0/g" input/${ts}.txt
		else
			echo "REPLACE decoder 1,1 WITH decoder 2,2 IN input/${ts}.txt"
			sed -i '' "s/decoder 1,1/decoder 2,2/g" input/${ts}.txt
			echo "REPLACE dcfraction ${refalpha} WITH dcfraction ${alpha} IN input/${ts}.txt"
			sed -i '' "s/dcfraction ${refalpha}/dcfraction ${alpha}/g" input/${ts}.txt
		fi

		# echo "REPLACE ecc Steane WITH ecc Steane,Steane IN input/${ts}.txt"
		# sed -i '' "s/ecc Steane/ecc Steane,Steane/g" input/${ts}.txt

		echo "xxxxxxx"
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
		echo "zipping ${outdir}/${ts}"
		tar -zcvf ${outdir}/${ts}.tar.gz ${outdir}/${ts}
	done
elif [[ "$1" == "to_cluster" ]]; then
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "Sending output ${ts} to cluster"
		scp -r /Users/pavi/Documents/chbank/$ts pavi@cedar.computecanada.ca:/project/def-jemerson/chbank/
		echo "Sending input file ${ts}.txt to cluster"
		scp /Users/pavi/Dropbox/rclearn/chflow/input/$ts.txt pavi@cedar.computecanada.ca:/project/def-jemerson/pavi/chflow/input/
		echo "Sending input file schedule_${ts}.txt to cluster"
		scp /Users/pavi/Dropbox/rclearn/chflow/input/schedule_$ts.txt pavi@cedar.computecanada.ca:/project/def-jemerson/pavi/chflow/input/
	done
elif [[ "$1" == "from_cluster" ]]; then
	for (( t=0; t<${#timestamps[@]}; ++t )); do
		ts=${timestamps[t]}
		echo "Bringing output ${ts} from cluster"
		scp -r pavi@cedar.computecanada.ca:/project/def-jemerson/chbank/$ts.tar.gz /Users/pavi/Documents/chbank
		tar -xvf /Users/pavi/Documents/chbank/$ts.tar.gz
		echo "Bringing input file ${ts}.txt from cluster"
		scp pavi@cedar.computecanada.ca:/project/def-jemerson/pavi/chflow/input/$ts.txt /Users/pavi/Dropbox/rclearn/chflow/input
		echo "Bringing schedule_${ts}.txt from cluster"
		scp pavi@cedar.computecanada.ca:/project/def-jemerson/pavi/chflow/input/schedule_$ts.txt /Users/pavi/Dropbox/rclearn/chflow/input
		echo "/project/def-jemerson WITH /Users/pavi/Documents IN input/${ts}.txt"
		sed -i '' "s/\/project\/def-jemerson/\Users\/pavi\/Documents/g" input/${ts}.txt
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
