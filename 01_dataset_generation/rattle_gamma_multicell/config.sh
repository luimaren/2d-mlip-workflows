# shellcheck shell=bash
# config.sh: lo único que hay que editar.

CUENTA="<LUMI_PROJECT>"                        # proyecto de LUMI
PSEUDO_DIR="/scratch/<LUMI_PROJECT>/pseudos"   # carpeta con B/C.pbe-n-kjpaw_psl.1.0.0.UPF
PW_EXE="pw.x"

cargar_qe() {   # las mismas líneas que el job.sh del strain
  module load LUMI/24.03 partition/C
  module load QuantumESPRESSO/7.5-cpeGNU-24.03
}

# Recursos por celda. min = minutos por caso; por = casos seguidos en cada elemento del array;
# sim = elementos a la vez. Con este reparto entran 177 jobs en small y 100 en standard,
# por debajo de los límites de LUMI (220 y 120 por usuario), así que todo se envía de una vez.
# 4x5 y 5x5 necesitan más de los 4 nodos que permite small (QE estima ~600 y ~850 GB).
# celda partición  nodos tareas/nodo mem  min por sim
RECURSOS="
1x1   small        1   16  32G    5  25   4
2x2   small        1   56  112G  10  10  10
2x3   small        1  112  224G  20   5  10
2x4   small        1  112  224G  30   4  10
3x3   small        2  112  224G  40   4  10
3x4   small        2  112  224G  60   4  10
3x5   small        3  112  224G  75   3   8
4x4   small        4  112  224G  75   3   8
4x5   standard     6  128  0     120  2   6
5x5   standard     8  128  0     150  2   6
"

export CUENTA PSEUDO_DIR PW_EXE RECURSOS
