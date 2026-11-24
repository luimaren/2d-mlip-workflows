#!/bin/bash
# Lanza con sbatch todos los casos (o solo los de una temperatura).
#   ./submit_all.sh          -> todos
#   ./submit_all.sh 2750     -> solo T2750K
cd "$(dirname "$0")" || exit 1
for d in T${1:-*}K/*/; do
    if [ -f "$d/job_BSC.sh" ]; then
        echo "Enviando $d"
        ( cd "$d" && sbatch job_BSC.sh )
    fi
done
