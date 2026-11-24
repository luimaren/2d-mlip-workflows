#!/bin/bash
# Envia los scf pendientes con sbatch (--cases, --max N, --dry-run, --force)
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON:-python3}" workflow.py submit "$@"
