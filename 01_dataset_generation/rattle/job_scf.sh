#!/bin/bash
#SBATCH --job-name=scf
#SBATCH --partition=small
#SBATCH --nodes=2
#SBATCH --ntasks=224
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=100G
#SBATCH --account=<LUMI_PROJECT>
#SBATCH --output=/dev/null

# Un job por caso (job array). Desde la carpeta del proyecto:  sbatch --array=1-200%20 job_scf.sh
cd cases/case_$(printf %03d $SLURM_ARRAY_TASK_ID) || exit 1
exec > qe_out.out 2> qe_out.err

export SLURM_CPU_BIND=none
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export EBU_USER_PREFIX=/project/<LUMI_PROJECT>/EasyBuild
module load LUMI/25.03
module load partition/C
module load QuantumESPRESSO/7.5-cpeGNU-25.03-CPU

srun pw.x -input input_scf.in
