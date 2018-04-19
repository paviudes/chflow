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
	cd src
	if [[ "$nd" = "null" ]]; then
		python remote.py ${ts}
	else
		python remote.py ${ts} ${nd}
	fi
	cd ..
fi