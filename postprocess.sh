# Copy input/cg1d_dc_0.001.txt to input/cg1d_dc_<alpha>.txt
alphas=("0" "0.001" "0.01" "0.1" "1")
# alphas=("0" "1" "2" "3")
rm input/temp.txt
touch input/temp.txt
echo -e "#\\n"
for dc in "${alphas[@]}"
do
	#sed -i 's/cg1d\/random/cg1d\/random\/knr/g' input/nc_cptp_dc_${dc}.txt
	# sed -i 's/pavi\//pavi\/Documents\/IQC\//g' input/nc_cptp_dc_${dc}.txt
	# sed -i 's/\/regen/\/regen\/wide/g' input/nc_cptp_dc_${dc}.txt
	echo -e "sbload nc_cptp_dc_${dc}\\npmetrics infid\\ncollect" >> input/temp.txt;
done
echo -e "exit" >> input/temp.txt
./chflow.sh -- temp.txt
