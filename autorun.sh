#!/bin/bash
./run_partial.sh generate
./run_partial.sh overwrite
parallel --joblog partial_decoders.log ./chflow.sh {1} :::: input/partial_decoders.txt
./run_partial.sh pmetrics
for i in {1..7}
do
	echo -ne '\007'
	sleep 1
done
