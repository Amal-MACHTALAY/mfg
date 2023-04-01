#!/bin/sh

#SBATCH -p himem  #compute
#SBATCH --qos=himem-cpu
#SBATCH -N 1
#SBATCH -n 56
#SBATCH --mem=0g
#SBATCH --job-name=case_name
#SBATCH --time=30:00:00
#SBATCH --exclusive

module purge
module load MUMPS
module load Python/3.8.2-GCCcore-9.3.0 CMake/3.16.4-GCCcore-9.3.0 OpenMPI/4.0.3-GCC-9.3.0 iccifort/2020.1.217

mpirun python mfg_parallel_main.py -snes_linesearch_type l2 
