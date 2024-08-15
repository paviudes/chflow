#./chflow.sh cptp_split
#./chflow.sh cptp_dp
./chflow.sh -- gather.txt 1
echo "DC 0"
cat ./../chbank/cptp/regen/cptp_dp/results/log_infid.txt
echo "DC 0.01"
cat ./../chbank/cptp/regen/cptp_split/results/log_infid.txt
echo "DC 0.1"
cat ./../chbank/cptp/regen/cptp_split_0.1/results/log_infid.txt
