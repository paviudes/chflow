#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=12:00:00

#SBATCH --array=0-1:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=48
#SBATCH --nodes=1
#SBATCH --output=alpha_0.1_%A_%a.out

#SBATCH -o /project/def-jemerson/chbank/16_07_2020_18_16_04/results/ouptut_%j.o
#SBATCH -e /project/def-jemerson/chbank/16_07_2020_18_16_04/results/errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

module load intel/2016.4 python/3.7.0 scipy-stack/2019a
cd /project/def-jemerson/pavi/chflow
./chflow.sh 16_07_2020_18_16_04 ${SLURM_ARRAY_TASK_ID}
