#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=2:00:00

#SBATCH --array=0-0:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --output=rtz_soft_%A_%a.out

#SBATCH -o /home/pavi/projects/def-jemerson/pavi/chbank/12_02_2019_10_41_36/results/ouptut_%j.o
#SBATCH -e /home/pavi/projects/def-jemerson/pavi/chbank/12_02_2019_10_41_36/results/errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

module load gcc/7.3.0 python/3.7.0 scipy-stack/2019a
cd /home/pavi/projects/def-jemerson/pavi/chflow
./chflow.sh 12_02_2019_10_41_36 ${SLURM_ARRAY_TASK_ID}
