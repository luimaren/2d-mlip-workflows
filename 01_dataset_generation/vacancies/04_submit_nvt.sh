#!/bin/bash
# Envia (sbatch) las NVT preparadas.  Ej: ./04_submit_nvt.sh --cases 12 --temps 1000
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON:-python3}" scripts/workflow.py submit-nvt "$@"
