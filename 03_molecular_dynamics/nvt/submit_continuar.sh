#!/bin/bash
#SBATCH --job-name=md_bgdy_10x10_1500
#SBATCH --account=<LUMI_PROJECT>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --dependency=singleton
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Continua la MD de md_nvt.py desde el ultimo fotograma de md_bgdy_10x10_1500_1fs.traj.
# Uso (desde la carpeta de la MD, con md_nvt_continuar.py y el .model):
#     mkdir -p logs && sbatch submit_continuar.sh
# --dependency=singleton: los trabajos con este mismo nombre se ejecutan de uno en uno,
# asi que se pueden enviar varios seguidos y cada uno continua donde paro el anterior.
# small-g admite hasta 3 dias; con --time menor el script se adapta solo.

set -euo pipefail

module purge
module load Local-LAIF lumi-aif-singularity-bindings

export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-torch-u24r70f21m50t210-20260807_115122.sif
export VENV=$HOME/mace_test_env
export OMP_NUM_THREADS=7

cd "${SLURM_SUBMIT_DIR}"

# Hora a la que Slurm cortara este trabajo (para que Python pare antes, en un fotograma guardado)
MD_FIN_EPOCH="${SLURM_JOB_END_TIME:-}"
if [ -z "${MD_FIN_EPOCH}" ]; then
    fin=$(squeue -h -j "${SLURM_JOB_ID}" -o %e 2>/dev/null || true)
    if [ -n "${fin}" ]; then
        MD_FIN_EPOCH=$(date -d "${fin}" +%s 2>/dev/null || true)
    fi
fi
echo "Fin previsto del trabajo: ${MD_FIN_EPOCH:-desconocido} (epoch)"
export MD_FIN_EPOCH

srun singularity exec "${SIF}" \
    bash -c "source ${VENV}/bin/activate && python -u md_nvt_continuar.py"
