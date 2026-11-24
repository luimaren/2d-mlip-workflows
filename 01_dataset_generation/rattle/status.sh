#!/bin/bash
# Casos sin terminar (sin convergencia o sin JOB DONE). Reenvío:  sbatch --array=<lista> job_scf.sh
ids=$(for d in cases/case_*; do
  grep -qs "convergence has been achieved" $d/qe_out.out && grep -qs "JOB DONE" $d/qe_out.out || echo $((10#${d##*_}))
done | paste -sd,)
echo "sin terminar: ${ids:-ninguno}"
