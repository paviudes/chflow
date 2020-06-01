#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=5:00:00

#SBATCH --ntasks-per-node=48
#SBATCH --nodes=1
#SBATCH -o /Users/pavi/Documents/chbank/01_06_2020_15_15_38/results/post_%j.o
#SBATCH -e /Users/pavi/Documents/chbank/01_06_2020_15_15_38/results/post_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

module load intel/2016.4 python/3.7.0 scipy-stack/2019a
cd /project/def-jemerson/pavi/chflow
./chflow -- post_01_06_2020_15_15_38.txt
cd /Users/pavi/Documents/chbank
tar -zcvf 01_06_2020_15_15_38.tar.gz 01_06_2020_15_15_38
