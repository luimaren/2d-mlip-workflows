#!/bin/bash
# Un elemento del array: scf de pw.x (Γ) sobre los casos de su línea de la lista. Lo lanza enviar.sh.
set -u
source "$SLURM_SUBMIT_DIR/config.sh"
cargar_qe
export OMP_NUM_THREADS=1 ESPRESSO_PSEUDO="$PSEUDO_DIR"

for d in $(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1"); do
  cd "$d" || continue
  grep -qs "JOB DONE" output.out && continue
  echo "[$(date +%T)] $d"
  srun "$PW_EXE" -nk 2 -in input.in > output.out 2> error.err &
  pid=$!
  # pw.x tiene que arrancar como un solo programa MPI; si salen copias en serie, se cancela el job
  for _ in $(seq 24); do
    sleep 5
    kill -0 $pid 2>/dev/null || break
    if [ "$(grep -c 'Program PWSCF' output.out)" -gt 1 ]; then
      echo "ERROR: pw.x no ha arrancado con MPI (revisa cargar_qe)"; kill $pid; exit 1
    fi
  done
  wait $pid
  grep -qs "JOB DONE" output.out && rm -rf tmp
done
