#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=1-0:00:00

#SBATCH --array=0-0:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=40
#SBATCH --nodes=1
#SBATCH --output=mcrand_power_%A_%a.out

#SBATCH -o /project/def-jemerson/chbank/12_06_2020_11_15_30/results/ouptut_%j.o
#SBATCH -e /project/def-jemerson/chbank/12_06_2020_11_15_30/results/errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

module load intel python scipy-stack
cd /project/def-jemerson/pavi/chflow
./chflow.sh 12_06_2020_11_15_30 ${SLURM_ARRAY_TASK_ID}
