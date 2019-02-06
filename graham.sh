#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=1:00:00

#SBATCH --array=0-0:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --mem=31744
#SBATCH --output=pauli_%A_%a.out

#SBATCH -o /home/pavi/projects/def-jemerson/pavi/chbank/06_02_2019_12_16_39/results/ouptut_%j.o
#SBATCH -e /home/pavi/projects/def-jemerson/pavi/chbank/06_02_2019_12_16_39/results/errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

module load gcc/7.3.0 python/3.7.0 scipy-stack/2019a
cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
./chflow.sh 06_02_2019_12_16_39 ${SLURM_ARRAY_TASK_ID}
