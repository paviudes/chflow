#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=5:00:00

#SBATCH --ntasks-per-node=48
#SBATCH --nodes=1
#SBATCH -o /project/def-jemerson/chbank/16_07_2020_18_17_53/results/pre_ouptut_%j.o
#SBATCH -e /project/def-jemerson/chbank/16_07_2020_18_17_53/results/pre_errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

module load intel/2016.4 python/3.7.0 scipy-stack/2019a
cd /project/def-jemerson/pavi/chflow
./chflow.sh -- cedar/pre_16_07_2020_18_17_53.txt
