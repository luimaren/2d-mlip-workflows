#!/bin/bash
# Crea cases/<caso>/relax/ (input QE + job + pseudos) a partir de structures/*.xyz
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON:-python3}" scripts/workflow.py prepare "$@"
