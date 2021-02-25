./chflow.sh -- infid_gen.txt
# Change the code in the rpZrpX file
sed -i'.backup' -e "s/rep7Z,rep7X/rep7X,rep7Z/g" input/vary_infid_rpXrpZ.txt
sed -i'.backup' -e "s/rep7Z,rep7X/rep7X,rep7X/g" input/vary_infid_rpXrpX.txt
sed -i'.backup' -e "s/rep7Z,rep7X/rep7Z,rep7Z/g" input/vary_infid_rpZrpZ.txt
