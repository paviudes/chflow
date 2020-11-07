#!/bin/bash
host=$(hostname)
echo "Host: $host"

if [[ $host == *"paviws"* ]]; then
	outdir="/Users/pavi/Documents"
	chflowdir="/Users/pavi/Dropbox/rclearn/chflow"
	cores=$(sysctl -n hw.ncpu)
	sed_prepend="'' "
	# nonpauli_timestamps=("21_07_2020_16_28_27" "20_07_2020_20_11_41" "21_07_2020_21_49_52" "03_08_2020_22_47_51" "03_08_2020_23_03_35" "03_08_2020_22_54_43" "04_08_2020_18_31_52" "04_08_2020_18_49_45" "04_08_2020_19_00_34" "21_07_2020_16_28_30" "25_08_2020_01_24_08" "25_08_2020_01_24_23")
	pauli_timestamps=("10_08_2020_00_33_27" "10_08_2020_00_33_34" "10_08_2020_00_33_35" "10_08_2020_00_33_36" "10_08_2020_10_04_35" "10_08_2020_10_04_37" "10_08_2020_16_53_00" "10_08_2020_16_53_01" "10_08_2020_16_53_02" "10_08_2020_16_53_03" "10_08_2020_16_53_04" "10_08_2020_16_53_05")
	# alphas=(0 0.0001 0.0003 0.0005 0.0007 0.0012 0.0025 0.0036 0.005 0.01 0.02 1)
	# ising_level3_timestamps=("29_10_2020_13_17_31" "29_10_2020_13_17_32" "29_10_2020_13_17_33" "29_10_2020_13_17_34" "29_10_2020_13_17_35" "29_10_2020_13_17_36" "29_10_2020_13_17_37" "29_10_2020_13_17_38" "29_10_2020_13_17_39" "29_10_2020_13_17_40" "29_10_2020_13_17_41" "29_10_2020_13_17_42")
	ising_timestamps=("29_10_2020_13_16_31" "29_10_2020_13_16_32" "29_10_2020_13_16_33" "29_10_2020_13_16_34" "29_10_2020_13_16_35" "29_10_2020_13_16_36" "29_10_2020_13_16_37" "29_10_2020_13_16_38" "29_10_2020_13_16_39" "29_10_2020_13_16_40" "29_10_2020_13_16_41" "29_10_2020_13_16_42" "29_10_2020_13_16_43" "29_10_2020_13_16_44" "29_10_2020_13_16_45" "29_10_2020_13_16_46" "29_10_2020_13_16_47" "29_10_2020_13_16_48" "29_10_2020_13_16_49" "29_10_2020_13_16_50" "29_10_2020_13_16_51" "29_10_2020_13_16_52" "29_10_2020_13_16_53" "29_10_2020_13_16_54" "29_10_2020_13_16_55" "29_10_2020_13_16_56" "29_10_2020_13_16_57" "29_10_2020_13_16_58" "29_10_2020_13_16_59" "29_10_2020_13_16_60" "29_10_2020_13_16_61" "29_10_2020_13_16_62" "29_10_2020_13_16_63" "29_10_2020_13_16_64" "29_10_2020_13_16_65" "29_10_2020_13_16_66" "29_10_2020_13_16_67" "29_10_2020_13_16_68" "29_10_2020_13_16_69" "29_10_2020_13_16_70" "29_10_2020_13_16_71" "29_10_2020_13_16_72" "29_10_2020_13_16_73" "29_10_2020_13_16_74" "29_10_2020_13_16_75" "29_10_2020_13_16_76" "29_10_2020_13_16_77" "29_10_2020_13_16_78")
	ising_level3_timestamps=("29_10_2020_13_17_31" "29_10_2020_14_17_34" "29_10_2020_14_17_36" "29_10_2020_14_17_38" "29_10_2020_14_17_40" "29_10_2020_14_17_42" "29_10_2020_14_17_44" "29_10_2020_14_17_50" "29_10_2020_14_17_53" "29_10_2020_14_17_63" "29_10_2020_14_17_66" "29_10_2020_14_17_78")
	ising_level3_ext_timestamps=("29_10_2020_13_17_31" "29_10_2020_13_17_34" "29_10_2020_13_17_36" "29_10_2020_13_17_38" "29_10_2020_13_17_40" "29_10_2020_13_17_42" "29_10_2020_13_17_44" "29_10_2020_13_17_50" "29_10_2020_13_17_53" "29_10_2020_13_17_63" "29_10_2020_13_17_66" "29_10_2020_13_17_78")
	nonpauli_timestamps=("29_10_2020_13_17_31" "30_10_2020_13_17_31" "30_10_2020_13_17_32" "30_10_2020_13_17_33" "30_10_2020_13_17_34" "30_10_2020_13_17_35" "30_10_2020_13_17_36" "30_10_2020_13_17_37" "30_10_2020_13_17_38" "30_10_2020_13_17_39" "30_10_2020_13_17_40" "30_10_2020_13_17_41" "30_10_2020_13_17_42" "30_10_2020_13_17_43" "30_10_2020_13_17_44" "30_10_2020_13_17_45" "30_10_2020_13_17_46" "30_10_2020_13_17_47" "30_10_2020_13_17_48" "30_10_2020_13_17_49" "30_10_2020_13_17_50" "30_10_2020_13_17_51" "30_10_2020_13_17_52" "30_10_2020_13_17_53" "30_10_2020_13_17_54" "30_10_2020_13_17_55" "30_10_2020_13_17_56" "30_10_2020_13_17_57" "30_10_2020_13_17_58" "30_10_2020_13_17_59" "30_10_2020_13_17_60" "30_10_2020_13_17_61" "30_10_2020_13_17_62" "30_10_2020_13_17_63" "30_10_2020_13_17_64" "30_10_2020_13_17_65" "30_10_2020_13_17_66" "30_10_2020_13_17_67" "30_10_2020_13_17_68" "30_10_2020_13_17_69" "30_10_2020_13_17_70" "30_10_2020_13_17_71" "30_10_2020_13_17_72" "30_10_2020_13_17_73" "30_10_2020_13_17_74" "30_10_2020_13_17_75" "30_10_2020_13_17_76" "30_10_2020_13_17_77" "30_10_2020_13_17_78")
	
	ising_level3_imp_final_timestamps=("29_10_2020_13_16_31" "29_10_2020_13_17_32" "29_10_2020_13_17_34" "29_10_2020_13_17_36" "29_10_2020_13_17_38" "29_10_2020_13_17_40" "29_10_2020_13_17_42" "29_10_2020_13_17_44" "29_10_2020_13_17_50" "29_10_2020_13_17_53" "29_10_2020_13_17_63" "29_10_2020_13_17_66" "29_10_2020_13_17_78" "31_10_2020_13_17_54" "32_10_2020_13_17_54")
    ising_level3_dir_final_timestamps=("29_10_2021_13_17_31" "29_10_2021_13_17_32" "29_10_2021_13_17_34" "29_10_2021_13_17_36" "29_10_2021_13_17_38" "29_10_2021_13_17_40" "29_10_2021_13_17_42" "29_10_2021_13_17_44" "29_10_2021_13_17_50" "29_10_2021_13_17_53" "29_10_2021_13_17_63" "29_10_2021_13_17_66" "29_10_2021_13_17_78" "31_10_2021_13_17_54")
    ## Cluster runs
    alphas=(0 0.0001 0.00013 0.00021 0.00027 0.00035 0.00044 0.00093 0.00135 0.00326 0.00368 0.00467 0.00678 1)

	# alphas=(0 0.0001 0.00011 0.00013 0.00015 0.00016 0.00019 0.00021 0.00024 0.00027 0.00031 0.00035 0.00039 0.00044 0.0005  0.00057 0.00064 0.00073 0.00082 0.00093 0.00105 0.00119 0.00135 0.00153 0.00173 0.00196 0.00222 0.00251 0.00284 0.00322 0.00364 0.00413 0.00467 0.00529 0.00599 0.00678 0.00767 0.00868 0.00983 0.01113 0.01259 0.01426 0.01614 0.01827 0.02068 0.02341 0.0265 1)
	# alphas=(0.00153 0.00173 0.00196 0.00222 0.00251 0.00284 0.00322 0.00364 0.00413)
	# alphas=(0 0.00326 0.00329 0.00333 0.00337 0.0034 0.00344 0.00348 0.00352 0.00356 0.0036)
	# alphas=(0.00368 0.00376 0.00385 0.00394 0.00402 0.00411 0.00421 0.0043 0.0044 0.0045)
	# alphas=(0 0.00375 0.00391 0.00407 0.00424 0.00442 0.0046)

	# All range
	# alphas=(0 0.00013 0.00021 0.00027 0.00035 0.00044 0.00093 0.00135 0.00326 0.00368 0.00467 0.00678)
	# 21 to 52
	# alphas=(0 0.0013 0.00142 0.00156 0.0017 0.00186 0.00204 0.00223 0.00244 0.00267 0.00292 0.0032)
	# 51 to 61.
	# alphas=(0 0.00311 0.00317 0.00323 0.0033 0.00336 0.00342 0.00348 0.00354 0.0036 0.00366 0.00372)

elif [[ $host == "oem-ThinkPad-X1-Carbon-Gen-8" ]]; then
	outdir="/home/oem/Documents"
	chflowdir="/home/oem/Desktop/Research_PhD/chflow"
	cores=$(nproc --all)
	nonpauli_timestamps=("21_07_2020_16_28_27" "03_08_2020_22_54_43" "04_08_2020_18_31_52" "04_08_2020_18_49_45" "04_08_2020_19_00_34" "21_07_2020_16_28_30" "25_08_2020_01_24_08" "25_08_2020_01_24_23")
	pauli_timestamps=("10_08_2020_00_33_27" "10_08_2020_10_04_37" "10_08_2020_16_53_00" "10_08_2020_16_53_01" "10_08_2020_16_53_02" "10_08_2020_16_53_03" "10_08_2020_16_53_04" "10_08_2020_16_53_05")
	alphas=(0 0.0001 0.0003 0.0007 0.0012 0.005 0.02 1)
else
	outdir="/project/def-jemerson/chbank"
	chflowdir="/project/def-jemerson/${USER}/chflow"
	nonpauli_timestamps=("30_10_2020_13_17_31" "30_10_2020_13_17_32" "30_10_2020_13_17_33" "30_10_2020_13_17_34" "30_10_2020_13_17_35" "30_10_2020_13_17_36" "30_10_2020_13_17_37" "30_10_2020_13_17_38" "30_10_2020_13_17_39" "30_10_2020_13_17_40" "30_10_2020_13_17_41" "30_10_2020_13_17_42" "30_10_2020_13_17_43" "30_10_2020_13_17_44" "30_10_2020_13_17_45" "30_10_2020_13_17_46" "30_10_2020_13_17_47" "30_10_2020_13_17_48" "30_10_2020_13_17_49" "30_10_2020_13_17_50" "30_10_2020_13_17_51" "30_10_2020_13_17_52" "30_10_2020_13_17_53" "30_10_2020_13_17_54" "30_10_2020_13_17_55" "30_10_2020_13_17_56" "30_10_2020_13_17_57" "30_10_2020_13_17_58" "30_10_2020_13_17_59" "30_10_2020_13_17_60" "30_10_2020_13_17_61" "30_10_2020_13_17_62" "30_10_2020_13_17_63" "30_10_2020_13_17_64" "30_10_2020_13_17_65" "30_10_2020_13_17_66" "30_10_2020_13_17_67" "30_10_2020_13_17_68" "30_10_2020_13_17_69" "30_10_2020_13_17_70" "30_10_2020_13_17_71" "30_10_2020_13_17_72" "30_10_2020_13_17_73" "30_10_2020_13_17_74" "30_10_2020_13_17_75" "30_10_2020_13_17_76" "30_10_2020_13_17_77" "30_10_2020_13_17_78")
	ising_timestamps=("29_10_2020_13_16_31" "29_10_2020_13_16_32" "29_10_2020_13_16_33" "29_10_2020_13_16_34" "29_10_2020_13_16_35" "29_10_2020_13_16_36" "29_10_2020_13_16_37" "29_10_2020_13_16_38" "29_10_2020_13_16_39" "29_10_2020_13_16_40" "29_10_2020_13_16_41" "29_10_2020_13_16_42" "29_10_2020_13_16_43" "29_10_2020_13_16_44" "29_10_2020_13_16_45" "29_10_2020_13_16_46" "29_10_2020_13_16_47" "29_10_2020_13_16_48" "29_10_2020_13_16_49" "29_10_2020_13_16_50" "29_10_2020_13_16_51" "29_10_2020_13_16_52" "29_10_2020_13_16_53" "29_10_2020_13_16_54" "29_10_2020_13_16_55" "29_10_2020_13_16_56" "29_10_2020_13_16_57" "29_10_2020_13_16_58" "29_10_2020_13_16_59" "29_10_2020_13_16_60" "29_10_2020_13_16_61" "29_10_2020_13_16_62" "29_10_2020_13_16_63" "29_10_2020_13_16_64" "29_10_2020_13_16_65" "29_10_2020_13_16_66" "29_10_2020_13_16_67" "29_10_2020_13_16_68" "29_10_2020_13_16_69" "29_10_2020_13_16_70" "29_10_2020_13_16_71" "29_10_2020_13_16_72" "29_10_2020_13_16_73" "29_10_2020_13_16_74" "29_10_2020_13_16_75" "29_10_2020_13_16_76" "29_10_2020_13_16_77" "29_10_2020_13_16_78")
	cores=48
	ising_level3_timestamps=("29_10_2020_13_17_31" "29_10_2020_13_17_32" "29_10_2020_13_17_33" "29_10_2020_13_17_34" "29_10_2020_13_17_35" "29_10_2020_13_17_36" "29_10_2020_13_17_37" "29_10_2020_13_17_38" "29_10_2020_13_17_39" "29_10_2020_13_17_40" "29_10_2020_13_17_41" "29_10_2020_13_17_42" "29_10_2020_13_17_43" "29_10_2020_13_17_44" "29_10_2020_13_17_45" "29_10_2020_13_17_46" "29_10_2020_13_17_47" "29_10_2020_13_17_48" "29_10_2020_13_17_49" "29_10_2020_13_17_50" "29_10_2020_13_17_51" "29_10_2020_13_17_52" "29_10_2020_13_17_53" "29_10_2020_13_17_54" "29_10_2020_13_17_55" "29_10_2020_13_17_56" "29_10_2020_13_17_57" "29_10_2020_13_17_58" "29_10_2020_13_17_59" "29_10_2020_13_17_60" "29_10_2020_13_17_61" "29_10_2020_13_17_62" "29_10_2020_13_17_63" "29_10_2020_13_17_64" "29_10_2020_13_17_65" "29_10_2020_13_17_66" "29_10_2020_13_17_67" "29_10_2020_13_17_68" "29_10_2020_13_17_69" "29_10_2020_13_17_70" "29_10_2020_13_17_71" "29_10_2020_13_17_72" "29_10_2020_13_17_73" "29_10_2020_13_17_74" "29_10_2020_13_17_75" "29_10_2020_13_17_76" "29_10_2020_13_17_77" "29_10_2020_13_17_78")
	# ising_level3_imp_final_timestamps=("29_10_2020_13_17_31" "29_10_2020_13_17_32" "29_10_2020_13_17_34" "29_10_2020_13_17_36" "29_10_2020_13_17_38" "29_10_2020_13_17_40" "29_10_2020_13_17_42" "29_10_2020_13_17_44" "29_10_2020_13_17_50" "29_10_2020_13_17_53" "29_10_2020_13_17_63" "29_10_2020_13_17_66" "29_10_2020_13_17_78" "31_10_2020_13_17_54")
	ising_level3_imp_final_timestamps=("29_10_2020_13_16_31" "29_10_2020_13_17_32" "29_10_2020_13_17_34" "29_10_2020_13_17_36" "29_10_2020_13_17_38" "29_10_2020_13_17_40" "29_10_2020_13_17_42" "29_10_2020_13_17_44" "29_10_2020_13_17_50" "29_10_2020_13_17_53" "29_10_2020_13_17_63" "29_10_2020_13_17_66" "29_10_2020_13_17_78" "31_10_2020_13_17_54" "32_10_2020_13_17_54")
    ising_level3_dir_final_timestamps=("29_10_2021_13_17_31" "29_10_2021_13_17_32" "29_10_2021_13_17_34" "29_10_2021_13_17_36" "29_10_2021_13_17_38" "29_10_2021_13_17_40" "29_10_2021_13_17_42" "29_10_2021_13_17_44" "29_10_2021_13_17_50" "29_10_2021_13_17_53" "29_10_2021_13_17_63" "29_10_2021_13_17_66" "29_10_2021_13_17_78" "31_10_2021_13_17_54")
	# alphas=(0 0.0001 0.00011 0.00013 0.00015 0.00016 0.00019 0.00021 0.00024 0.00027 0.00031 0.00035 0.00039 0.00044 0.0005  0.00057 0.00064 0.00073 0.00082 0.00093 0.00105 0.00119 0.00135 0.00153 0.00173 0.00196 0.00222 0.00251 0.00284 0.00322 0.00364 0.00413 0.00467 0.00529 0.00599 0.00678 0.00767 0.00868 0.00983 0.01113 0.01259 0.01426 0.01614 0.01827 0.02068 0.02341 0.0265 1)
	alphas=(0 0.0001 0.00013 0.00021 0.00027 0.00035 0.00044 0.00093 0.00135 0.00326 0.00368 0.00467 0.00678 1)

	sed_prepend=""
fi

# alphas=(0 0.0001 0.00011 0.00013 0.00015 0.00016 0.00019 0.00021 0.00024 0.00027 0.00031 0.00035 0.00039 0.00044 0.0005  0.00057 0.00064 0.00073 0.00082 0.00093 0.00105 0.00119 0.00135 0.00153 0.00173 0.00196 0.00222 0.00251 0.00284 0.00322 0.00364 0.00413 0.00467 0.00529 0.00599 0.00678 0.00767 0.00868 0.00983 0.01113 0.01259 0.01426 0.01614 0.01827 0.02068 0.02341 0.0265 1)

rerun() {
	echo "removing ${outdir}/$1/channels/*"
	rm ${outdir}/$1/channels/*
	rm ${outdir}/$1/metrics/*
	rm ${outdir}/$1/results/*.npy
	# rm ${outdir}/chbank/$1/physical/*.npy
	echo "Running $1"
	echo "${ts}" >> input/partial_decoders.txt
}

display() {
	./chflow.sh $ts
}

timestamps=("${ising_level3_imp_final_timestamps[@]}")

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

		echo -e "\033[2mremoving ${outdir}/${ts}/physical/*\033[0m"
		rm ${outdir}/${ts}/physical/*

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
		trash ${outdir}/chbank/$ts
		echo "Bringing output ${ts} from cluster"
		scp -r pavi@cedar.computecanada.ca:/project/def-jemerson/chbank/$ts.tar.gz ${outdir}/chbank
		cd ${outdir}/chbank
		tar -xvf $ts.tar.gz
		cd ${chflowdir}
		#### Bring from chflow
		# echo "Bringing input file ${ts}.txt from cluster"
		# scp pavi@cedar.computecanada.ca:/project/def-jemerson/pavi/chflow/input/$ts.txt ${chflowdir}/input
		# echo "Bringing schedule_${ts}.txt from cluster"
		# scp pavi@cedar.computecanada.ca:/project/def-jemerson/pavi/chflow/input/schedule_$ts.txt ${chflowdir}/input
		#### Bring from def-jemerson/input
		echo "Bringing input file ${ts}.txt from cluster"
		scp pavi@cedar.computecanada.ca:/project/def-jemerson/input/$ts.txt ${chflowdir}/input
		echo "Bringing schedule_${ts}.txt from cluster"
		scp pavi@cedar.computecanada.ca:/project/def-jemerson/input/schedule_$ts.txt ${chflowdir}/input
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
