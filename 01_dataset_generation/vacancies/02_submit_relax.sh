#!/bin/bash
# Envia (sbatch) el relax de los casos pendientes.  Ej: ./02_submit_relax.sh --cases 1-8 [--dry-run]
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON:-python3}" scripts/workflow.py submit-relax "$@"
