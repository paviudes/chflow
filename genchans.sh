#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --begin=now
#SBATCH --time=2:00:00

#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --mem=31744

#SBATCH -o /project/def-jemerson/chbank/genchans/pcorr/ouptut_%j.o
#SBATCH -e /project/def-jemerson/chbank/genchans/pcorr/errors_%j.o

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

module load intel/2016.4 python/3.7.0 scipy-stack/2019a
cd /project/def-jemerson/pavi/chflow
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
./chflow.sh -- pcorrchans.txt
