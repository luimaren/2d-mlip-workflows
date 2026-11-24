#!/bin/bash
#SBATCH --job-name=md_bgdy15x15
#SBATCH --account=<LUMI_PROJECT>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Uso (desde la carpeta con md_nvt.py, FTv30-96.model y bgdy_15x15.xyz):
#     mkdir -p logs && sbatch submit_md.sh

set -euo pipefail

module purge
module load Local-LAIF lumi-aif-singularity-bindings

export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-torch-u24r70f21m50t210-20260807_115122.sif
export VENV=$HOME/mace_test_env
export OMP_NUM_THREADS=7

cd "${SLURM_SUBMIT_DIR}"

srun singularity exec "${SIF}" \
    bash -c "source ${VENV}/bin/activate && python -u md_nvt.py"
