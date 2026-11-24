#!/bin/bash

#SBATCH --job-name=qe_run
#SBATCH --partition=gpp
#SBATCH --nodes=2
#SBATCH --ntasks=224  # Number of MPI ranks per node
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1     # Number of OpenMP threads per MPI rank
#SBATCH --time=72:00:00
#SBATCH --output=output.1
#SBATCH --error=error.1         # Specify the number of GPUs to be used for each node
#SBATCH --account=<BSC_ACCOUNT>        # Specify the account
#SBATCH --qos=gp_ehpc          # Define the appropriate QoS

#SBATCH --mail-type=all
#SBATCH --mail-user=<your_email>

# Load required modules

module purge
module load intel mkl impi fftw quantumespresso/7.5


# Set environment variables
export SLURM_CPU_BIND=none
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run QuantumESPRESSO simula
srun pw.x -input input.in
