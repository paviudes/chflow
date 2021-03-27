### Generate the input files
touch input/bias_gen.txt
echo "sbload vary_bias_stst" >> input/bias_gen.txt
echo "submit vary_bias_stst" >> input/bias_gen.txt
echo "sbload vary_bias_cycy" >> input/bias_gen.txt
./chflow.sh -- bias_gen.txt
# Change the code in the ststst file
sed -i'.backup' -e "s/Steane/7qc_cyclic/g" input/vary_bias_cycy.txt
#
### Run simulations
/chflow.sh vary_bias_stst
/chflow.sh vary_bias_cycy
#
### Compute metrics
touch input/bias_metrics.txt
echo "sbload vary_bias_stst" >> input/bias_metrics.txt
echo "pmetrics infid" >> input/bias_metrics.txt
echo "sbload vary_bias_cycy" >> input/bias_metrics.txt
echo "pmetrics infid" >> input/bias_metrics.txt
./chflow.sh -- bias_metrics.txt
#
### Plot results
touch input/bias_plot.txt
./chflow.sh -- bias_plot.txt
