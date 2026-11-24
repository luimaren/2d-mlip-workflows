#!/bin/bash
# relax2aimd por caso -> cases/<caso>/nvt_T<T>K/ listo para lanzar + xyz al dataset/
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON:-python3}" scripts/workflow.py postprocess "$@"
