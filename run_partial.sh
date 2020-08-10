#!/bin/bash
rerun() {
  echo "removing /Users/pavi/Documents/chbank/$1/channels/*"
  rm /Users/pavi/Documents/chbank/$1/channels/*
  rm /Users/pavi/Documents/chbank/$1/metrics/*
  rm /Users/pavi/Documents/chbank/$1/results/*.npy
  echo "Running $1"
  ./chflow.sh $ts
}

display() {
  ./chflow.sh $ts
}

pauli_timestamps=("21_07_2020_16_28_27" "20_07_2020_20_11_41" "21_07_2020_21_49_52" "03_08_2020_22_47_51" "03_08_2020_23_03_35" "03_08_2020_22_54_43" "04_08_2020_18_31_52" "04_08_2020_18_49_45" "04_08_2020_19_00_34" "21_07_2020_16_28_30")

cptp_timestamps=("06_08_2020_19_31_31" "06_08_2020_19_31_32" "06_08_2020_19_31_34" "06_08_2020_19_31_35" "06_08_2020_19_31_37" "06_08_2020_19_31_49" "06_08_2020_19_31_50" "06_08_2020_19_31_52" "06_08_2020_19_31_53" "06_08_2020_19_31_55")

rtasu_timestamps=("07_08_2020_00_44_26" "07_08_2020_00_44_27" "07_08_2020_00_44_29" "07_08_2020_00_44_30" "07_08_2020_00_44_31" "07_08_2020_00_44_33" "07_08_2020_00_44_36" "07_08_2020_00_44_38" "07_08_2020_00_44_39" "07_08_2020_00_44_40")

# nonpauli_timestamps=("10_08_2020_00_33_27" "10_08_2020_00_33_29" "10_08_2020_00_33_30" "10_08_2020_00_33_31" "10_08_2020_00_33_33" "10_08_2020_00_33_34" "10_08_2020_00_33_35" "10_08_2020_00_33_36" "10_08_2020_10_04_35" "10_08_2020_10_04_37" "10_08_2020_10_04_38" "10_08_2020_10_04_39" "10_08_2020_10_04_41" "10_08_2020_10_04_42" "10_08_2020_10_04_43" "10_08_2020_10_04_44" "10_08_2020_10_04_52" "10_08_2020_10_04_53" "10_08_2020_00_33_38" "10_08_2020_00_33_39")

nonpauli_timestamps=("10_08_2020_00_33_27" "10_08_2020_00_33_29" "10_08_2020_00_33_30" "10_08_2020_00_33_31" "10_08_2020_00_33_33" "10_08_2020_00_33_34" "10_08_2020_00_33_35" "10_08_2020_00_33_36" "10_08_2020_10_04_35" "10_08_2020_10_04_37" "10_08_2020_10_04_38" "10_08_2020_10_04_39")

timestamps=("${nonpauli_timestamps[@]}")
# b=("${a[@]}")

alphas=(0 0.0001 0.0002 0.0003 0.0004 0.0007 0.0012 0.0019 0.0031 0.005 1 "ML")
# alphas=(0.0022 0.0025 0.0027 0.03 0.0032 0.0035 0.0037 0.04 0.0042 0.0045)
# timestamps=("04_08_2020_19_00_34")
if [[ "$1" == "overwrite" ]]; then
  for (( t=0; t<${#timestamps[@]}; ++t )); do
    ts=${timestamps[t]}
    rerun $ts &
    echo "xxxxxxx"
  done
  wait
elif [[ "$1" == "generate" ]]; then
  refts=${timestamps[0]}
	refalpha=${alphas[0]}
  for (( t=1; t<${#timestamps[@]}; ++t )); do
    ts=${timestamps[t]}

    echo -e "\033[2mremoving /Users/pavi/Documents/chbank/${ts}/physical/*\033[0m"
    rm /Users/pavi/Documents/chbank/${ts}/physical/*

		alpha=${alphas[t]}

    echo "alpha = ${alpha}"
    echo "sbload ${refts}" > input/temp.txt
		echo "submit ${ts}" >> input/temp.txt
		echo "quit" >> input/temp.txt
		cat input/temp.txt
    ./chflow.sh -- temp.txt
    rm input/temp.txt

    # oldnrate=15
    # newnrate=15
    # oldfrac=0.3
    # newfrac=0.95
    #
		# echo "REPLACE noiserange $oldnrate,$oldnrate,1;2,2,1;$oldfrac,$oldfrac,1;0.1,0.1,1 WITH noiserange $newnrate,$newnrate,1;2,2,1;$newfrac,$newfrac,1;0.1,0.1,1 IN input/${ts}.txt"
		# sed -i '' "s/noiserange $oldnrate,$oldnrate,1;2,2,1;$oldfrac,$oldfrac,1;0.1,0.1,1/noiserange $newnrate,$newnrate,1;2,2,1;$newfrac,$newfrac,1;0.1,0.1,1/" input/${ts}.txt
    #
		# echo "REPLACE $oldnrate WITH $newnrate IN input/schedule_${ts}.txt"
		# sed -i '' "s/$oldnrate/$newnrate/" input/schedule_${ts}.txt
    # echo "REPLACE $oldfrac WITH $newfrac IN input/schedule_${ts}.txt"
		# sed -i '' "s/$oldfrac/$newfrac/" input/schedule_${ts}.txt

    # oldnrate=6
    # newnrate=8
    #
		# echo "REPLACE noiserange $oldnrate,$oldnrate,1;1,1,1 WITH noiserange $newnrate,$newnrate,1;1,1,1 IN input/${ts}.txt"
		# sed -i '' "s/noiserange $oldnrate,$oldnrate,1;1,1,1/noiserange $newnrate,$newnrate,1;1,1,1/" input/${ts}.txt
    #
		# echo "REPLACE $oldnrate WITH $newnrate IN input/schedule_${ts}.txt"
		# sed -i '' "s/$oldnrate/$newnrate/" input/schedule_${ts}.txt

    # oldp=0.02
    # newp=0.02
    # oldangle=0.02
    # newangle=0.01
    #
		# echo "REPLACE noiserange $oldp,$oldp,1;$oldangle,$oldangle,1; WITH noiserange $newp,$newp,1;$newangle,$newangle,1 IN input/${ts}.txt"
		# sed -i '' "s/noiserange $oldp,$oldp,1;$oldangle,$oldangle,1/noiserange $newp,$newp,1;$newangle,$newangle,1/" input/${ts}.txt
    #
		# echo "REPLACE $oldp $oldangle WITH $newp $newangle IN input/schedule_${ts}.txt"
		# sed -i '' "s/$oldp $oldangle/$newp $newangle/" input/schedule_${ts}.txt

    if [[ "$alpha" == "ML" ]]; then
      echo "REPLACE decoder 1,1 WITH decoder 0,0 IN input/${ts}.txt"
  		sed -i '' "s/decoder 1,1/decoder 0,0/g" input/${ts}.txt
    else
      echo "REPLACE decoder 1,1 WITH decoder 2,2 IN input/${ts}.txt"
  		sed -i '' "s/decoder 1,1/decoder 2,2/g" input/${ts}.txt
      echo "REPLACE dcfraction ${refalpha} WITH dcfraction ${alpha} IN input/${ts}.txt"
  		sed -i '' "s/dcfraction ${refalpha}/dcfraction ${alpha}/g" input/${ts}.txt
    fi

    # if [[ "$alpha" == "ML" ]]; then
    #   echo "REPLACE decoder 1 WITH decoder 0 IN input/${ts}.txt"
  	# 	sed -i '' "s/decoder 1/decoder 0/g" input/${ts}.txt
    # else
    #   echo "REPLACE decoder 1 WITH decoder 2 IN input/${ts}.txt"
  	# 	sed -i '' "s/decoder 1/decoder 2/g" input/${ts}.txt
    #   echo "REPLACE dcfraction ${refalpha} WITH dcfraction ${alpha} IN input/${ts}.txt"
  	# 	sed -i '' "s/dcfraction ${refalpha}/dcfraction ${alpha}/g" input/${ts}.txt
    # fi

    # echo "REPLACE [100000] WITH [1000000] IN input/${ts}.txt"
		# sed -i '' "s/[100000]/[1000000]/g" input/${ts}.txt

		echo "xxxxxxx"
  done
else
  for (( t=0; t<${#timestamps[@]}; ++t )); do
    ts=${timestamps[t]}
    display $ts
    echo "xxxxxxx"
  done
fi
