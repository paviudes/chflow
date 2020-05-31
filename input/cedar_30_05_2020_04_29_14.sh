#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=12:00:00

#SBATCH --array=0-4:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=48
#SBATCH --nodes=1
#SBATCH --output=pcorr30_%A_%a.out

#SBATCH -o /project/def-jemerson/chbank/30_05_2020_04_29_14/results/ouptut_%j.o
#SBATCH -e /project/def-jemerson/chbank/30_05_2020_04_29_14/results/errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

module load intel/2016.4 python/3.7.0 scipy-stack/2019a
cd /project/def-jemerson/pavi/chflow
./chflow.sh 30_05_2020_04_29_14 ${SLURM_ARRAY_TASK_ID}
