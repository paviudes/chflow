#!/bin/bash
#SBATCH --account=def-emerson
#SBATCH --begin=now
#SBATCH --time=2-0:00:00

#SBATCH --array=0-9:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --mem=31744
#SBATCH --output=twirled_%A_%a.out

#SBATCH -o /Users/pavi/Documents/chbank/24_04_2020_16_48_48/results/ouptut_%j.o
#SBATCH -e /Users/pavi/Documents/chbank/24_04_2020_16_48_48/results/errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

cd /home/pavi/projects/def-jemerson/pavi/chflow
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
./chflow.sh 24_04_2020_16_48_48 ${SLURM_ARRAY_TASK_ID}
