#!/bin/bash
#SBATCH --account=default
#SBATCH --begin=now
#SBATCH --time=0:00:00

#SBATCH --array=0-4:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=40
#SBATCH --nodes=1
#SBATCH --output=X_%A_%a.out

#SBATCH -o /Users/pavi/Documents/chbank/11_06_2020_13_32_35/results/ouptut_%j.o
#SBATCH -e /Users/pavi/Documents/chbank/11_06_2020_13_32_35/results/errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=X

module load intel/2016.4 python/3.7.0 scipy-stack/2019a
cd /project/def-jemerson/$USER/chflow
./chflow.sh 11_06_2020_13_32_35 ${SLURM_ARRAY_TASK_ID}
