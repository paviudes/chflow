#!/bin/bash
parallel --joblog partial_decoders.log ./chflow.sh {1} :::: input/partial_decoders.txt
for i in {1..7}
do
	echo -ne '\007'
	sleep 1
done

