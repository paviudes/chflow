./chflow.sh -- infid_gen.txt
# Change the code in the cycy file
sed -i'.backup' -e "s/7qc_cyclic/5qc/g" input/vary_infid_fqfq.txt
sed -i'.backup' -e "s/7qc_cyclic,7qc_cyclic/rep5,7qc_cyclic/g" input/vary_infid_rpcy.txt
