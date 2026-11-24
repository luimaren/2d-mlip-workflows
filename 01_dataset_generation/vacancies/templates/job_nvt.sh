#!/bin/bash

#SBATCH --job-name=@JOB_NAME@
#SBATCH --partition=small
#SBATCH --nodes=2
#SBATCH --ntasks=224  # Number of MPI ranks per node
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1     # Number of OpenMP threads per MPI rank
#SBATCH --time=@TIME@
#SBATCH --mem=100G              # Allocate 224GB memory on each standard node
#SBATCH --output=@OUT@
#SBATCH --error=@ERR@           # Specify the number of GPUs to be used for each node
#SBATCH --account=<LUMI_PROJECT>        # Specify the account

# Set environment variables

export SLURM_CPU_BIND=none
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


# Load required modules

export EBU_USER_PREFIX=/project/<LUMI_PROJECT>/EasyBuild
module load LUMI/25.03
module load partition/C
module load QuantumESPRESSO/7.5-cpeGNU-25.03-CPU

# Run QuantumESPRESSO simula
srun pw.x -input @INPUT@

