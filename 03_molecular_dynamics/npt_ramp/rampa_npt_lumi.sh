#!/bin/bash
#SBATCH --job-name=rampa_npt
#SBATCH --account=<LUMI_PROJECT>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --signal=USR1@900
#
# Rampa térmica de BGDY 3x3 con MACE-MH-1, Langevin NPT en el plano (rampa_npt_bgdy.py).
#
#   mkdir -p logs
#   sbatch --partition=dev-g --time=00:30:00 rampa_npt_lumi.sh prueba   # prueba corta: ms/paso y horas
#   sbatch rampa_npt_lumi.sh                                             # relajación + equilibrado + rampa
#
# Modo por defecto (todo): hace lo que falte y reanuda lo que esté a medias.
# 15 min antes del límite de tiempo SLURM manda SIGUSR1: el script termina el bloque en curso, guarda un
# frame y sale con código 3. Entonces este trabajo se reenvía solo para seguir (como mucho MAX_REENVIOS veces).
set -euo pipefail

MODO=${1:-todo}
MAX_REENVIOS=5
REENVIO=${REENVIO:-0}

module load Local-LAIF lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-torch-u24r70f21m50t210-20260807_115122.sif
VENV=$HOME/mace_test_env
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-7}

cd "$SLURM_SUBMIT_DIR"
echo "Trabajo $SLURM_JOB_ID | modo $MODO | reenvío $REENVIO de $MAX_REENVIOS | $(hostname) | $(date)"

# exec: python sustituye a bash y recibe directamente la señal de SLURM
codigo=0
srun singularity exec "$SIF" bash -c "source $VENV/bin/activate && exec python -u rampa_npt_bgdy.py $MODO" || codigo=$?

echo "Fin del trabajo $SLURM_JOB_ID: código $codigo | $(date)"
if [ "$codigo" -eq 3 ] && [ "$MODO" = "todo" ]; then
    if [ "$REENVIO" -lt "$MAX_REENVIOS" ]; then
        sbatch --export=ALL,REENVIO=$((REENVIO + 1)) rampa_npt_lumi.sh todo
        echo "Sin terminar: reenviado para seguir (reenvío $((REENVIO + 1)) de $MAX_REENVIOS)."
    else
        echo "Alcanzado MAX_REENVIOS: para seguir, sbatch rampa_npt_lumi.sh"
    fi
fi
exit "$codigo"
