#!/bin/bash
# Seguimiento desde el nodo de login (no usa la GPU ni carga el modelo):
#   bash ver_npt.sh            últimas líneas de progreso del equilibrado y de la rampa
#   bash ver_npt.sh figuras    figuras y análisis (PNG y analisis.txt en la carpeta de cada etapa)
#   bash ver_npt.sh figuras --semillas 1 2 3
set -euo pipefail
MODO=${1:-estado}
shift || true

module load Local-LAIF lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-torch-u24r70f21m50t210-20260807_115122.sif
VENV=$HOME/mace_test_env

singularity exec "$SIF" bash -c "source $VENV/bin/activate && python rampa_npt_bgdy.py $MODO $*"
