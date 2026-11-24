#!/bin/bash
# Envía un array de Slurm por celda con los casos pendientes.
#   ./enviar.sh              casos sin output.out que no se hayan enviado ya
#   ./enviar.sh --reenviar   todos los que no tienen "JOB DONE" (fallidos o cortados); con la cola vacía
#   ./enviar.sh --prueba     enseña los sbatch sin enviar
set -eu
cd "$(dirname "$0")"
RAIZ=$PWD
source ./config.sh
MODO=${1:-}
mkdir -p listas logs

echo "$RECURSOS" | while read -r celda part nodos tareas mem min por sim; do
  [ -z "$celda" ] && continue
  lista="$RAIZ/listas/$celda.$(date +%Y%m%d-%H%M%S).txt"
  : > "$lista"
  n=0; linea=""
  for d in casos/"$celda"/s*; do
    [ -d "$d" ] || continue
    grep -qs "JOB DONE" "$d/output.out" && continue
    if [ "$MODO" != --reenviar ] && { [ -f "$d/output.out" ] || [ -f "$d/enviado" ]; }; then continue; fi
    linea="$linea $RAIZ/$d"; n=$((n + 1))
    if (( n % por == 0 )); then echo "$linea" >> "$lista"; linea=""; fi
  done
  [ -n "$linea" ] && echo "$linea" >> "$lista"
  ntar=$(wc -l < "$lista")
  if [ "$ntar" -eq 0 ]; then rm -f "$lista"; continue; fi

  cmd=(sbatch --parsable --account="$CUENTA" --partition="$part" --job-name="rat_$celda"
       --nodes="$nodos" --ntasks-per-node="$tareas" --cpus-per-task=1 --mem="$mem"
       --time=$((min * por + 5)) --array="1-$ntar%$sim" --output="$RAIZ/logs/%x_%A_%a.out"
       "$RAIZ/job.sh" "$lista")
  echo "$celda: $n casos en $ntar tareas"
  if [ "$MODO" = --prueba ]; then echo "   ${cmd[*]}"; rm -f "$lista"; continue; fi
  jid=$("${cmd[@]}")
  for d in $(cat "$lista"); do echo "$jid" > "$d/enviado"; done
  echo "   job $jid"
done
