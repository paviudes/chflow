#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=3-0:00:00

#SBATCH --array=0-13:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=48
#SBATCH --nodes=1
#SBATCH --output=rtas106_%A_%a.out

#SBATCH -o /Users/pavi/Documents/chbank/12_06_2020_22_26_12/results/ouptut_%j.o
#SBATCH -e /Users/pavi/Documents/chbank/12_06_2020_22_26_12/results/errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=2003adityajain@gmail.com

module load intel/2016.4 python/3.7.0 scipy-stack/2019a
cd /project/def-jemerson/$USER/chflow
./chflow.sh 12_06_2020_22_26_12 ${SLURM_ARRAY_TASK_ID}
