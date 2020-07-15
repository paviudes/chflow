#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=20:00:00

#SBATCH --array=0-1:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=40
#SBATCH --nodes=1
#SBATCH --output=mcrtas_zoomed_%A_%a.out

#SBATCH -o /Users/pavi/Documents/chbank/26_06_2020_20_07_38/results/ouptut_%j.o
#SBATCH -e /Users/pavi/Documents/chbank/26_06_2020_20_07_38/results/errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=2003adityajain@gmail.com

module load intel/2016.4 python/3.7.0 scipy-stack/2019a
cd /project/def-jemerson/$USER/chflow
./chflow.sh 26_06_2020_20_07_38 ${SLURM_ARRAY_TASK_ID}
