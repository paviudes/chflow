#!/bin/bash
rerun() {
  echo "removing /Users/pavi/Documents/chbank/$1/channels/*"
  rm /Users/pavi/Documents/chbank/$1/channels/*
  echo "Running $1"
  ./chflow.sh $ts
}

display() {
  ./chflow.sh $ts
}

pauli_timestamps=("21_07_2020_16_28_27" "20_07_2020_20_11_41" "21_07_2020_21_49_52" "03_08_2020_22_47_51" "03_08_2020_23_03_35" "03_08_2020_22_54_43" "04_08_2020_18_31_52" "04_08_2020_18_49_45" "04_08_2020_19_00_34" "21_07_2020_16_28_30")

cptp_timestamps=("06_08_2020_19_31_31" "06_08_2020_19_31_32" "06_08_2020_19_31_34" "06_08_2020_19_31_35" "06_08_2020_19_31_37" "06_08_2020_19_31_49" "06_08_2020_19_31_50" "06_08_2020_19_31_52" "06_08_2020_19_31_53" "06_08_2020_19_31_55")

timestamps=("${cptp_timestamps[@]}")
# b=("${a[@]}")

alphas=(0 0.00009 0.00016 0.00029 0.00052 0.00093 0.00167 0.003 1 "ML")
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
    echo "sbload ${refts}" > input/temp.txt
		echo "submit ${ts}" >> input/temp.txt
		echo "quit" >> input/temp.txt
		cat input/temp.txt
    ./chflow.sh -- temp.txt
    rm input/temp.txt

    # oldnrate=20
    # newnrate=15
    #
		# echo "REPLACE noiserange $oldnrate,$oldnrate,1;2,2,1;0.3,0.3,1;0.1,0.1,1 WITH noiserange $newnrate,$newnrate,1;2,2,1;0.3,0.3,1;0.1,0.1,1 IN input/${ts}.txt"
		# sed -i '' "s/noiserange $oldnrate,$oldnrate,1;2,2,1;0.3,0.3,1;0.1,0.1,1/noiserange $newnrate,$newnrate,1;2,2,1;0.3,0.3,1;0.1,0.1,1/" input/${ts}.txt

		# echo "REPLACE $oldnrate WITH $newnrate IN input/schedule_${ts}.txt"
		# sed -i '' "s/15/$newnrate/" input/schedule_${ts}.txt

    if [[ "$alpha" == "ML" ]]; then
      echo "REPLACE decoder 1,1 WITH decoder 0,0 IN input/${ts}.txt"
  		sed -i '' "s/decoder 1,1/decoder 0,0/" input/${ts}.txt
    else
      echo "REPLACE decoder 1,1 WITH decoder 2,2 IN input/${ts}.txt"
  		sed -i '' "s/decoder 1,1/decoder 2,2/" input/${ts}.txt
      echo "REPLACE dcfraction ${refalpha} WITH dcfraction ${alpha} IN input/${ts}.txt"
  		sed -i '' "s/dcfraction ${refalpha}/dcfraction ${alpha}/" input/${ts}.txt
    fi

    # echo "REPLACE direct WITH power IN input/${ts}.txt"
		# sed -i '' "s/direct/power/" input/${ts}.txt

		echo "xxxxxxx"
  done
else
  for (( t=0; t<${#timestamps[@]}; ++t )); do
    ts=${timestamps[t]}
    display $ts
    echo "xxxxxxx"
  done
fi
