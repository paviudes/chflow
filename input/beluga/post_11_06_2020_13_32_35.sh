#!/bin/bash
#SBATCH --account=default
#SBATCH --begin=now
#SBATCH --time=5:00:00

#SBATCH --ntasks-per-node=40
#SBATCH --nodes=1
#SBATCH -o /Users/pavi/Documents/chbank/11_06_2020_13_32_35/results/post_%j.o
#SBATCH -e /Users/pavi/Documents/chbank/11_06_2020_13_32_35/results/post_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=X

module load intel/2016.4 python/3.7.0 scipy-stack/2019a
cd /project/def-jemerson/$USER/chflow
./chflow -- post_11_06_2020_13_32_35.txt
cd /Users/pavi/Documents/chbank
tar -zcvf 11_06_2020_13_32_35.tar.gz 11_06_2020_13_32_35
