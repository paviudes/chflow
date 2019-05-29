#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=16:00:00

#SBATCH --array=0-11:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --mem=31744
#SBATCH --output=rand_%A_%a.out

#SBATCH -o /home/pavi/projects/def-jemerson/pavi/chbank/27_05_2019_22_01_25/results/ouptut_%j.o
#SBATCH -e /home/pavi/projects/def-jemerson/pavi/chbank/27_05_2019_22_01_25/results/errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

cd /home/pavi/projects/def-jemerson/pavi/chflow
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
./chflow.sh 27_05_2019_22_01_25 ${SLURM_ARRAY_TASK_ID}
