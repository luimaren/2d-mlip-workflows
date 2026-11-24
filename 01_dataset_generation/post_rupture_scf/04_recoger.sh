#!/bin/bash
# Junta los SCF convergidos en extxyz para reentrenar MACE (dataset/):
#   postrotura_scf_todas.extxyz, _train.extxyz, _test.extxyz (10 % al azar), _control.extxyz,
#   _resumen.csv y _fuera.txt (los que no entran y por qué).
#   ./04_recoger.sh [--test 0.15] [--config-type postrotura_FT] [--salida dataset]
cd "$(dirname "$0")" || exit 1
exec python3 workflow.py recoger "$@"
