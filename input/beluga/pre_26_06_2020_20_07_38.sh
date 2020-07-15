#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=5:00:00

#SBATCH --ntasks-per-node=40
#SBATCH --nodes=1
#SBATCH -o /Users/pavi/Documents/chbank/26_06_2020_20_07_38/results/pre_ouptut_%j.o
#SBATCH -e /Users/pavi/Documents/chbank/26_06_2020_20_07_38/results/pre_errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=2003adityajain@gmail.com

module load intel/2016.4 python/3.7.0 scipy-stack/2019a
cd /project/def-jemerson/$USER/chflow
./chflow.sh -- beluga/pre_26_06_2020_20_07_38.txt
