#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=2-0:00:00

#SBATCH --array=0-4:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --mem=31744
#SBATCH --output=rand_cptp_%A_%a.out

#SBATCH -o /Users/pavi/Documents/chbank/24_05_2020_10_36_31/results/ouptut_%j.o
#SBATCH -e /Users/pavi/Documents/chbank/24_05_2020_10_36_31/results/errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

cd /home/pavi/projects/def-jemerson/pavi/chflow
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
./chflow.sh 24_05_2020_10_36_31 ${SLURM_ARRAY_TASK_ID}
