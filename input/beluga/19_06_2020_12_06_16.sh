#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=12:00:00

#SBATCH --array=0-7:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=40
#SBATCH --nodes=1
#SBATCH --output=pcorr_l3_imp_%A_%a.out

#SBATCH -o /project/def-jemerson/chbank/19_06_2020_12_06_16/results/ouptut_%j.o
#SBATCH -e /project/def-jemerson/chbank/19_06_2020_12_06_16/results/errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

module load intel python scipy-stack
cd /project/def-jemerson/pavi/chflow
./chflow.sh 19_06_2020_12_06_16 ${SLURM_ARRAY_TASK_ID}
