#!/bin/bash
#SBATCH --account=rac-2018-hpcg1742
#SBATCH --partition=reserved

#SBATCH --begin=now
#SBATCH --time=8:00:00

#SBATCH --array=0-20:1
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10g
#SBATCH --output=rand_%A_%a.out

#SBATCH -o /global/home/hpc4198/chbank/24_04_2018_15_34_40/results/ouptut_%j.o
#SBATCH -e /global/home/hpc4198/chbank/24_04_2018_15_34_40/results/errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

module load anaconda/2.7.13
module load gcc/6.4.0
cd $SLURM_SUBMIT_DIR
cd src/simulate/
python compile.py build_ext --inplace > compiler_output.txt 2>&1
cd $SLURM_SUBMIT_DIR
./chflow.sh 24_04_2018_15_34_40 ${SLURM_ARRAY_TASK_ID}
