# alphas=("0.00134" "0.01288" "0.07056" "0.24359" "0.55505" "0.86652" "1")
alphas=("0.05" "0.5")
rm runs.sh
touch runs.sh

for dc in "${alphas[@]}"
do
	
	# Check if the input file already exists for a particular alpha.
	if [ -e input/cg1d_dc_${dc}.txt ]
	then
		echo "Input file input/cg1d_dc_${dc}.txt already exists. Skipping alpha = $dc to prevent a rerun."
	
	else
		echo "Creating input files for alpha = $dc."

		# Create a text file named prep_alpha.txt with the following lines.
		# 1. sbload cg1d
		# 2. submit cg1d_dc_<value of the dc variable>
		# 3. exit

		echo -e "sbload cg1d\\nsubmit cg1d_dc_${dc}\\nexit" > input/temp.txt;

		# Run chflow with the above created input file.
		./chflow.sh -- temp.txt
		
		# An input file named cg1d_dc_<value of the dc variable> will be created.
		# Change line dcfraction 0 to dcfraction <value of the dc variable>
		sed -i 's/dcfraction 0/dcfraction '${dc}'/g' input/cg1d_dc_${dc}.txt

		# Add the simulation instruction to runs.sh
		echo -e "./chflow.sh cg1d_dc_${dc}" >> runs.sh;		
	fi
done