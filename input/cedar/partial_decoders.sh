#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --nodes=1
#SBATCH --cpus-per-task=21
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH -o /project/def-jemerson/chbank/partial_output.o
#SBATCH -e /project/def-jemerson/chbank/partial_errors.o
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com
module load intel python scipy-stack
cd /project/def-jemerson/pavi/chflow
# Read the parameters, placing each column to their respective argument
parallel -j $SLURM_CPUS_PER_TASK ./chflow.sh {1} :::: ./partial_cluster.txt
