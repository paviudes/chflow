./chflow.sh -- pauli.txt
sed -i 's/Steane/7qc_cyclic/g' input/vary_bias_cycy.txt
./chflow.sh vary_bias_stst
./chflow.sh vary_bias_cycy
./chflow.sh -- test.txt
./run_partial.sh compare_plot
