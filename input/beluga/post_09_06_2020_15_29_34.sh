#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=5:00:00

#SBATCH --ntasks-per-node=40
#SBATCH --nodes=1
#SBATCH -o /project/def-jemerson/chbank/09_06_2020_15_29_34/results/post_%j.o
#SBATCH -e /project/def-jemerson/chbank/09_06_2020_15_29_34/results/post_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

module load intel/2016.4 python/3.7.0 scipy-stack/2019a
cd /project/def-jemerson/pavi/chflow
./chflow -- post_09_06_2020_15_29_34.txt
cd /project/def-jemerson/chbank
tar -zcvf 09_06_2020_15_29_34.tar.gz 09_06_2020_15_29_34
