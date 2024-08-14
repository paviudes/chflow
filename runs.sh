./chflow.sh cptp_split
./chflow.sh cptp_dp
./chflow.sh -- gather.txt 1
echo "Splitting decoder"
cat ./../chbank/unitary/regen/cptp_split/results/log_infid.txt
echo "Depolarizing channel"
cat ./../chbank/unitary/regen/cptp_dp/results/log_infid.txt
