#!/bin/bash
pids=( `pgrep python` )
pgrep_args=""
for element in "${pids[@]}"
do
	# echo "${element}"
	pgrep_args="${pgrep_args} -pid ${element}"
done
echo -e "\033[2mRunning top${pgrep_args}\033[0m"
top ${pgrep_args}