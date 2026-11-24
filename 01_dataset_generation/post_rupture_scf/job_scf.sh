#!/bin/bash
# SCF de QE para varios casos seguidos (MN5-GPP). No lo lances a mano: usa ./02_enviar_scf.sh,
# que le pasa la lista de casos y el tiempo (LIMITE_S). A mano sería:
#   sbatch --export=ALL,LIMITE_S=28800 job_scf.sh case_001 case_002
#
# Por cada caso (en casos/<caso>/):
#   - si ya convergió, lo salta (salvo FORZAR=1, que lo repite desde cero);
#   - si el primer intento acabó con "convergence NOT achieved", hace solo el reintento;
#   - si no, ejecuta input.in y, si no converge, input_reintento.in (si queda tiempo);
#   - comprueba que pw.x arrancó con todos los procesos MPI (si no, para el trabajo: fallo de lanzamiento);
#   - borra tmp/ y apunta una línea en historial.txt.
# Antes de empezar un caso mira si queda tiempo (la duración del caso más largo de este trabajo, o EST_S);
# si no queda, lo deja para el siguiente envío.

#SBATCH --job-name=scf_postrot
#SBATCH --partition=gpp
#SBATCH --nodes=2
#SBATCH --ntasks=224
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/scf-%j.out
#SBATCH --error=logs/scf-%j.err
#SBATCH --account=<BSC_ACCOUNT>
#SBATCH --qos=gp_ehpc
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=<your_email>

set -u

module purge
module load intel mkl impi fftw quantumespresso/7.5

export SLURM_CPU_BIND=none
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

RAIZ=${SLURM_SUBMIT_DIR:-$PWD}
PW_FLAGS=${PW_FLAGS:--nk 2}     # pools sobre el espín (nspin = 2 en Γ: npool ≤ 2). "" para quitarlo
LIMITE_S=${LIMITE_S:-28800}     # duración pedida al trabajo (s); la pone 02_enviar_scf.sh
MARGEN_S=${MARGEN_S:-900}       # margen de seguridad antes del límite
EST_S=${EST_S:-3600}            # estimación de un caso hasta medir el primero
REINTENTO=${REINTENTO:-1}       # 1: si no converge, prueba input_reintento.in
FORZAR=${FORZAR:-0}             # 1: repite desde cero aunque ya hubiera convergido
NTASKS=${SLURM_NTASKS:-224}
T0=$(date +%s)

if ! command -v pw.x > /dev/null; then
    echo "ERROR: no encuentro pw.x tras cargar los módulos" >&2
    exit 1
fi
echo "Trabajo ${SLURM_JOB_ID:-local}: $# casos, ${NTASKS} procesos, PW_FLAGS='${PW_FLAGS}', límite ${LIMITE_S} s"
echo "Nodos: ${SLURM_JOB_NODELIST:-?}"

convergido() {   # $1 = output
    [ -f "$1" ] && grep -q "^!" "$1" && grep -q "convergence has been achieved" "$1" \
        && ! grep -q "convergence NOT achieved" "$1" && grep -q "JOB DONE" "$1"
}
no_convergido() { [ -f "$1" ] && grep -q "convergence NOT achieved" "$1"; }
mpi_ok() {       # una sola ejecución de pw.x y con todos los procesos
    local n np
    np=$(grep -c "Program PWSCF" "$1")
    n=$(grep -m1 -oE "(running on|Number of MPI processes:) +[0-9]+" "$1" | grep -oE "[0-9]+$")
    [ "$np" -le 1 ] && [ "${n:-0}" = "$NTASKS" ]
}
queda_tiempo() { [ $(( $(date +%s) - T0 + EST_S + MARGEN_S )) -le "$LIMITE_S" ]; }
ejecutar() {     # $1 = input, $2 = output (sin extensión)
    local t1 rc dur
    t1=$(date +%s)
    srun --ntasks="$NTASKS" pw.x $PW_FLAGS -input "$1" > "$2.out" 2> "$2.err"
    rc=$?
    dur=$(( $(date +%s) - t1 ))
    [ "$dur" -gt "$EST_S" ] && EST_S=$dur
    rm -rf tmp
    echo "$(date '+%F %T') job=${SLURM_JOB_ID:-local} ntasks=${NTASKS} input=$1 rc=${rc} dur=${dur}s" >> historial.txt
    echo "    $1 -> $2.out: rc=${rc}, ${dur} s"
    return $rc
}

for caso in "$@"; do
    dir="$RAIZ/casos/$caso"
    if [ ! -f "$dir/input.in" ]; then
        echo "$caso: no existe $dir/input.in"
        continue
    fi
    cd "$dir" || continue
    rm -f .en_cola

    if [ "$FORZAR" = 1 ]; then
        rm -f output.out output.err output_reintento.out output_reintento.err
    elif convergido output.out || convergido output_reintento.out; then
        echo "$caso: ya convergido"
        cd "$RAIZ"; continue
    fi
    if ! queda_tiempo; then
        echo "$caso: no queda tiempo (estimación ${EST_S} s); queda para el siguiente envío"
        cd "$RAIZ"; break
    fi
    echo "${SLURM_JOB_ID:-local}" > .en_marcha
    echo "$caso:"

    if no_convergido output.out && [ "$REINTENTO" = 1 ] && [ -f input_reintento.in ]; then
        etapa=reintento                       # el primer intento ya se hizo en un trabajo anterior
    else
        ejecutar input.in output
        if ! mpi_ok output.out; then
            echo "FALLO DE LANZAMIENTO MPI en $caso (pw.x no arrancó con ${NTASKS} procesos)." \
                 "Se para el trabajo; revisa logs/ y el entorno." >&2
            rm -f .en_marcha; cd "$RAIZ"; exit 2
        fi
        etapa=hecho
        if no_convergido output.out && [ "$REINTENTO" = 1 ] && [ -f input_reintento.in ]; then
            etapa=reintento
        fi
    fi
    if [ "$etapa" = reintento ]; then
        if queda_tiempo; then
            echo "    no convergió: reintento con input_reintento.in"
            ejecutar input_reintento.in output_reintento
            if ! mpi_ok output_reintento.out; then
                echo "FALLO DE LANZAMIENTO MPI en $caso (reintento)." >&2
                rm -f .en_marcha; cd "$RAIZ"; exit 2
            fi
        else
            echo "    no convergió y no queda tiempo para el reintento: queda para el siguiente envío"
        fi
    fi

    if convergido output.out || convergido output_reintento.out; then
        echo "    CONVERGIDO"
    else
        echo "    sin converger (ver ./03_estado.sh)"
    fi
    rm -f .en_marcha
    cd "$RAIZ"
done

echo "Fin del trabajo: $(( $(date +%s) - T0 )) s"
