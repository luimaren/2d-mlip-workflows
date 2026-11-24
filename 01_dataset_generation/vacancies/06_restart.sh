#!/bin/bash
# Reanuda calculos interrumpidos.  Ej: ./06_restart.sh relax   |   ./06_restart.sh nvt --cases 3
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON:-python3}" scripts/workflow.py restart "$@"
