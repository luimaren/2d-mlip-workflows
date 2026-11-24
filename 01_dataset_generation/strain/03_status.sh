#!/bin/bash
# Tabla de estado de los scf (squeue + contenido de qe_out.out)  (--cases)
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON:-python3}" workflow.py status "$@"
