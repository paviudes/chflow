#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --ntasks-per-node=48
#SBATCH -o /project/def-jemerson/chbank/partial_output.o
#SBATCH -e /project/def-jemerson/chbank/partial_errors.o
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com
module load intel python scipy-stack
cd /project/def-jemerson/pavi/chflow
parallel --joblog parallel.log ./chflow.sh {1} < partial_cluster.txt
