#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=5:00:00

#SBATCH --ntasks-per-node=48
#SBATCH --nodes=1
#SBATCH -o /Users/pavi/Documents/chbank/01_06_2020_15_11_34/results/pre_ouptut_%j.o
#SBATCH -e /Users/pavi/Documents/chbank/01_06_2020_15_11_34/results/pre_errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

module load intel/2016.4 python/3.7.0 scipy-stack/2019a
cd /project/def-jemerson/pavi/chflow
./chflow.sh -- cedar/pre_01_06_2020_15_11_34.txt
