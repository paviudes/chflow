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

timestamps=("20_07_2020_20_11_41" "21_07_2020_21_49_52" "03_08_2020_22_47_51" "03_08_2020_23_03_35" "03_08_2020_22_54_43" "04_08_2020_18_31_52" "04_08_2020_18_41_14" "04_08_2020_18_43_59" "04_08_2020_18_49_45" "04_08_2020_18_51_04" "04_08_2020_19_00_34")
# timestamps=("04_08_2020_19_00_34")
if [[ $1 -eq "overwrite" ]]; then
  for (( t=0; t<${#timestamps[@]}; ++t )); do
    ts=${timestamps[t]}
    rerun $ts &
    # if (( $t % 12 == 0)); then
    #   wait;
    # fi
    echo "xxxxxxx"
  done
  wait
fi

for (( t=0; t<${#timestamps[@]}; ++t )); do
  display $ts
done
