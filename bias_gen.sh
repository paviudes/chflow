./chflow.sh -- bias_gen.txt
# Change the code in the cycy file
sed -i'.backup' -e "s/Steane/7qc_cyclic/g" input/vary_bias_cycy.txt
sed -i'.backup' -e "s/Steane/5qc/g" input/vary_bias_fqfq.txt
sed -i'.backup' -e "s/Steane,Steane/rep5,7qc_cyclic/g" input/vary_bias_rpcy.txt
