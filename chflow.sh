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
	if [ "$BQMAMMOUTH" = "mp2" ] || [ "$BQMAMMOUTH" = "ms" ]; then
		if [[ "$BQMAMMOUTH" == "mp2" ]]; then
			module load gcc/5.2.0 intel64/14.0.0.080 PICOS/1.1.1 anaconda64/2.4.1.1
		else
			module load multiprocess/0.70.4.dev0 anaconda/2-4.3.0 gcc/4.8.1
		fi
		#####
		export MKL_NUM_THREADS=1
		cd src/simulate/
		python compile.py build_ext --inplace >> compiler_output.txt 2>&1
		cd ./../../
		#####
		cd src/
		if [[ "$nd" = "null" ]]; then
			python remote.py ${ts}
		else
			python remote.py ${ts} ${nd}
		fi
		cd ..
	else
		# No modules to load
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
fi