module load intel python scipy-stack
cd /project/def-jemerson/pavi/chflow
parallel --joblog partial_decoders.log --jobs  ./chflow.sh {1} :::: input/partial_decoders.txt
