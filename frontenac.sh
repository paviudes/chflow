#!/bin/bash
#SBATCH --account=default
#SBATCH --partition=standard

#SBATCH --begin=now
#SBATCH --time=1-0:00:00

#SBATCH --array=0-0:1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=rand_%A_%a.out

#SBATCH -o /global/home/hpc4198/chbank/20_04_2018_19_14_44/results/ouptut_%j.o
#SBATCH -e /global/home/hpc4198/chbank/20_04_2018_19_14_44/results/errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

module load anaconda/2.7.13
module load gcc/6.4.0
cd $SLURM_SUBMIT_DIR
cd src/simulate/
python compile.py build_ext --inplace > compiler_output.txt 2>&1
cd $SLURM_SUBMIT_DIR
./chflow.sh 20_04_2018_19_14_44 ${SLURM_ARRAY_TASK_ID}
