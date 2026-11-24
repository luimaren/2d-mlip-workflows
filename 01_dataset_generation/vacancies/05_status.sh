#!/bin/bash
# Tabla de estado de relax y NVT de cada caso
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON:-python3}" scripts/workflow.py status "$@"
