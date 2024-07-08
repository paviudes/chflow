# Copy input/cg1d_dc_0.001.txt to input/cg1d_dc_<alpha>.txt
alphas=("0" "0.001" "0.01" "0.1")
#alphas=("0.00134" "0.01288" "0.07056" "0.24359")
for dc in "${alphas[@]}"
do
	#cp input/cg1d_dc_0.001.txt input/cg1d_dc_${dc}.txt
	#sed -i 's/0.001/'${dc}'/g' input/cg1d_dc_${dc}.txt
	#cp input/schedule_cg1d_dc_0.001.txt input/schedule_cg1d_dc_${dc}.txt
	echo -e "sbload cg1d_dc_${dc}_dpfill\\npmetrics infid,unknown_errbg\\ncollect\\nexit" > input/temp.txt;
	./chflow.sh -- temp.txt
done
