#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=5:00:00

#SBATCH --ntasks-per-node=40
#SBATCH --nodes=1
#SBATCH -o /project/def-jemerson/chbank/12_06_2020_13_51_01/results/pre_ouptut_%j.o
#SBATCH -e /project/def-jemerson/chbank/12_06_2020_13_51_01/results/pre_errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

module load intel/2016.4 python/3.7.0 scipy-stack/2019a
cd /project/def-jemerson/pavi/chflow
./chflow.sh -- beluga/pre_12_06_2020_13_51_01.txt
