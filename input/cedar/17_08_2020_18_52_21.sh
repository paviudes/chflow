#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=8:00:00

#SBATCH --array=0-7:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=48
#SBATCH --nodes=1
#SBATCH --output=rtas_scat_imp_%A_%a.out

#SBATCH -o /project/def-jemerson/chbank/17_08_2020_18_52_21/results/ouptut_%j.o
#SBATCH -e /project/def-jemerson/chbank/17_08_2020_18_52_21/results/errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.srihar@gmail.com

module load intel python scipy-stack
cd /project/def-jemerson/pavi/chflow
./chflow.sh 17_08_2020_18_52_21 ${SLURM_ARRAY_TASK_ID}
