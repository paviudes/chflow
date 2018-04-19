#!/bin/bash
ts=${1:-null}
nd=${2:-null}
if [[ "$ts" = "null" ]]; then
	cd src
	python chflow.py
	cd ..
elif [[ "$ts" = "<" ]]; then
	cd src
	python chflow.py ${nd}
	cd ..
else
	# Modules for MP2

	# Modules for Frontenac
	module load anaconda/2.7.13
	module load gcc/6.4.0
	
	cd src/simulate/
	python compile.py build_ext --inplace > compiler_output.txt 2>&1
	cd ..
	if [[ "$nd" = "null" ]]; then
		python remote.py ${ts}
	else
		python remote.py ${ts} ${nd}
	fi
	cd ..
fi