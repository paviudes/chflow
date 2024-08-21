# Copy input/cg1d_dc_0.001.txt to input/cg1d_dc_<alpha>.txt
alphas=("0" "0.001" "0.01" "0.1" "1")
touch input/temp.txt
for dc in "${alphas[@]}"
do
	#cp input/cg1d_dc_0.001.txt input/cg1d_dc_${dc}.txt
	#sed -i 's/0.001/'${dc}'/g' input/cg1d_dc_${dc}.txt
	#cp input/schedule_cg1d_dc_0.001.txt input/schedule_cg1d_dc_${dc}.txt
	echo -e "sbload unitary_split_dc_${dc}\\npmetrics infid\\ncollect\\nexit" >> input/temp.txt;
done
./chflow.sh -- temp.txt
