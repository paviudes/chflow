#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=1-12:00:00

#SBATCH --array=0-9:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=40
#SBATCH --nodes=1
#SBATCH --output=cptp_l3_imp_%A_%a.out

#SBATCH -o /project/def-jemerson/chbank/19_06_2020_00_41_40/results/ouptut_%j.o
#SBATCH -e /project/def-jemerson/chbank/19_06_2020_00_41_40/results/errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

module load intel/2016.4 python/3.7.0 scipy-stack/2019a
cd /project/def-jemerson/pavi/chflow
./chflow.sh 19_06_2020_00_41_40 ${SLURM_ARRAY_TASK_ID}
