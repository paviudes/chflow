# Backup files by moving those are not required in future to an external hard drive.
external="/Volumes/PAVITHRAN/chbank/input"
future=("06_09_2017_21_32_03" "24_05_2020_23_33_22" "24_05_2020_23_25_57" "27_05_2020_15_55_18" "27_05_2020_18_47_06" "27_05_2020_16_01_59" "27_05_2020_13_24_40" "27_05_2020_16_06_43" "27_05_2020_19_26_15" "backup.sh" "sample_local" "sample_cluster" "test" "pcorrchans")

# Check membership in a list: https://stackoverflow.com/questions/14366390/check-if-an-element-is-present-in-a-bash-array

for file in *; do
  found=0
  for name in "${future[@]}"; do
    if [[ $file == *"$name"* ]]
    then
      found=1
      break
    fi
  done
  if [[ $found -eq 1 ]]
  then
    echo "Skipping $file for future use."
  else
    echo "\033[2mMoving $file to $external\033[0m"
    mv $file $external
  fi
done
