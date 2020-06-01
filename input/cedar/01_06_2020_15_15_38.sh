#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=5:00:00

#SBATCH --array=0-4:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=48
#SBATCH --nodes=1
#SBATCH --output=pcorr_%A_%a.out

#SBATCH -o /Users/pavi/Documents/chbank/01_06_2020_15_15_38/results/ouptut_%j.o
#SBATCH -e /Users/pavi/Documents/chbank/01_06_2020_15_15_38/results/errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

module load intel/2016.4 python/3.7.0 scipy-stack/2019a
cd /project/def-jemerson/pavi/chflow
./chflow.sh 01_06_2020_15_15_38 ${SLURM_ARRAY_TASK_ID}
