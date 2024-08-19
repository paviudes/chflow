./chflow.sh unitary_split
./chflow.sh unitary_split_0.01
./chflow.sh -- gather.txt 1
echo "DC 0"
cat ./../chbank/unitary/regen/unitary_split/results/log_infid.txt
echo "DC 0.01"
cat ./../chbank/unitary/regen/unitary_split_0.01/results/log_infid.txt
