#!/bin/bash
#SBATCH --job-name=mace_FTv32_sim7
#SBATCH --account=<LUMI_PROJECT>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# IMPORTANTE: la carpeta logs/ debe existir ANTES del sbatch
# (Slurm abre los ficheros de salida antes de ejecutar el script):
#     mkdir -p logs && sbatch submit_finetune.sh
#
# Config por defecto: config-FTv31-Naive.yml
#   (train/test en Dataset_BGDY3x3_v3/v1/, salida en MACE_models_BGDY3x3_FTv31/)
# Para usar otro:
#     sbatch --export=ALL,CONFIG=otro_config.yml submit_finetune.sh

set -euo pipefail

CONFIG="${CONFIG:-config-FTv31-Naive.yml}"

module purge
module load Local-LAIF lumi-aif-singularity-bindings

# Contenedor validado en la sesion interactiva (torch 2.10.0+rocm7.0).
# Si más adelante sale una version mas reciente, actualiza esta ruta.
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-torch-u24r70f21m50t210-20260807_115122.sif
export VENV=$HOME/mace_test_env

# Asegura que las rutas relativas del config (train_file, foundation_model, ...)
# se resuelvan desde el directorio donde se hizo el sbatch.
cd "${SLURM_SUBMIT_DIR}"

# --- Comprobaciones previas: falla rapido si falta algun fichero -------------
[[ -f "${CONFIG}" ]] || { echo "ERROR: no existe el config ${CONFIG}" >&2; exit 1; }

for key in train_file test_file foundation_model; do
    f=$(sed -nE "s/^${key}:[[:space:]]*\"?([^\"#]*)\"?.*/\1/p" "${CONFIG}" | head -n1 | xargs)
    if [[ -z "${f}" || ! -f "${f}" ]]; then
        echo "ERROR: ${key} -> '${f}' no encontrado (desde ${PWD})" >&2
        exit 1
    fi
    echo "OK ${key}: ${f}"
done
# -----------------------------------------------------------------------------

echo "Lanzando fine-tuning con config: ${CONFIG}"

srun singularity exec "${SIF}" \
    bash -c "source ${VENV}/bin/activate && python run_finetune.py ${CONFIG}"
