#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=5:00:00

#SBATCH --ntasks-per-node=48
#SBATCH --nodes=1
#SBATCH -o /Users/pavi/Documents/chbank/12_06_2020_22_26_12/results/post_%j.o
#SBATCH -e /Users/pavi/Documents/chbank/12_06_2020_22_26_12/results/post_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=2003adityajain@gmail.com

module load intel/2016.4 python/3.7.0 scipy-stack/2019a
cd /project/def-jemerson/$USER/chflow
./chflow -- post_12_06_2020_22_26_12.txt
cd /Users/pavi/Documents/chbank
tar -zcvf 12_06_2020_22_26_12.tar.gz 12_06_2020_22_26_12
