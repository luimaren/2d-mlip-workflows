#!/bin/bash
# Resumen de los SCF: cuántos han convergido, cuáles están en cola, cortados, sin converger o con error.
#   ./03_estado.sh              resumen
#   ./03_estado.sh --detalle    una línea por caso (iteraciones, minutos, E, magnetización)
#   ./03_estado.sh --csv        además escribe estado.csv
cd "$(dirname "$0")" || exit 1
exec python3 workflow.py estado "$@"
