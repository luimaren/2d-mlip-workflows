#!/bin/bash
# Envía los casos pendientes (sin empezar, cortados, sin reintento o con fallo MPI) en trabajos de K casos.
#   ./02_enviar_scf.sh --prueba          un caso en gp_debug (2 h) para medir cuánto tarda
#   ./02_enviar_scf.sh --dry-run         muestra los sbatch sin enviar
#   ./02_enviar_scf.sh                   todos los pendientes: 4 casos por trabajo, 8 h
#   ./02_enviar_scf.sh --k 6 --tiempo 12:00:00 --max-trabajos 20
#   ./02_enviar_scf.sh --casos case_007 case_031 [--forzar]
# Más opciones: ./02_enviar_scf.sh -h
cd "$(dirname "$0")" || exit 1
exec python3 workflow.py enviar "$@"
