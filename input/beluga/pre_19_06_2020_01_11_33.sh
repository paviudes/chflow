#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=3:00:00

#SBATCH --ntasks-per-node=40
#SBATCH --nodes=1
#SBATCH -o /project/def-jemerson/chbank/19_06_2020_01_11_33/results/pre_ouptut_%j.o
#SBATCH -e /project/def-jemerson/chbank/19_06_2020_01_11_33/results/pre_errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

module load intel python scipy-stack
cd /project/def-jemerson/pavi/chflow
./chflow.sh -- beluga/pre_19_06_2020_01_11_33.txt
