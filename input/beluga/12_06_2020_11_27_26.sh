#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=12:00:00

#SBATCH --array=0-2:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=40
#SBATCH --nodes=1
#SBATCH --output=mcpcorr_zoomed_%A_%a.out

#SBATCH -o /project/def-jemerson/chbank/12_06_2020_11_27_26/results/ouptut_%j.o
#SBATCH -e /project/def-jemerson/chbank/12_06_2020_11_27_26/results/errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

module load intel/2016.4 python/3.7.0 scipy-stack/2019a
cd /project/def-jemerson/pavi/chflow
./chflow.sh 12_06_2020_11_27_26 ${SLURM_ARRAY_TASK_ID}
