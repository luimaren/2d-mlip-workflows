#!/bin/bash
# Recopila los scf OK en dataset/: uniaxial_x, uniaxial_y, uniaxial, biaxial, aniso y total (.extxyz) + summary.csv  (--cases; necesita ASE)
# Los grupos (que modos van en cada .extxyz) se definen en GROUPS, al principio de workflow.py.
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON:-python3}" workflow.py collect "$@"
