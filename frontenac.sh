#!/bin/bash
#SBATCH --account=rac-2018-hpcg1742
#SBATCH --partition=reserved

#SBATCH --begin=now
#SBATCH --time=3:00:00

#SBATCH --array=0-20:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=24
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --output=rand_%A_%a.out

#SBATCH -o /global/home/hpc4198/chbank/27_04_2018_14_24_52/results/ouptut_%j.o
#SBATCH -e /global/home/hpc4198/chbank/27_04_2018_14_24_52/results/errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

module load anaconda/2.7.13
module load gcc/6.4.0
cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS = 1
export MKL_NUM_THREADS = 1
./chflow.sh 27_04_2018_14_24_52 ${SLURM_ARRAY_TASK_ID}
