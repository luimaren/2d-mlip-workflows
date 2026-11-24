#!/usr/bin/env python3
"""Láminas NxM de BGDY + rattle + inputs scf de pw.x en Γ.

    python3 genera.py relax_final.xyz

Escribe laminas/BGDY_<celda>.xyz (sin rattle) y casos/<celda>/s<sigma>_<idx>/input.in.
No toca los casos que ya tienen output.out. Requiere ASE.
"""
import os
import sys

import numpy as np
from ase.io import read, write

CELDAS = ['1x1', '2x2', '2x3', '2x4', '3x3', '3x4', '3x5', '4x4', '4x5', '5x5']
SIGMAS = [0.01, 0.03, 0.05, 0.07]     # Å, desviación de cada componente x, y, z (ase.Atoms.rattle)
N_POR_SIGMA = 25                       # 25 x 4 sigmas = 100 por celda -> 1000 en total
REP = 3                                # relax_final.xyz es la 3x3 de la celda primitiva
SEMILLA = 2026
CONV_THR_POR_ATOMO = 7.35e-6 / 126     # Ry: el conv_thr del strain (126 átomos), escalado por átomo

# Ajustes de QE (los del dataset de strain). Si tu input.in del strain difiere, cámbialo aquí.
QE = """&CONTROL
  calculation = 'scf'
  prefix = 'bgdy'
  outdir = './tmp'
  tstress = .true.
  tprnfor = .true.
  disk_io = 'nowf'
/
&SYSTEM
  ibrav = 0
  nat = {nat}
  ntyp = 2
  ecutwfc = 60
  ecutrho = 600
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.005
  nspin = 2
  starting_magnetization(1) = 0.1
  starting_magnetization(2) = 0.1
  vdw_corr = 'grimme-d3'
  nosym = .true.
/
&ELECTRONS
  conv_thr = {conv_thr:.3e}
  mixing_beta = 0.3
  electron_maxstep = 200
/
ATOMIC_SPECIES
B 10.811 B.pbe-n-kjpaw_psl.1.0.0.UPF
C 12.0107 C.pbe-n-kjpaw_psl.1.0.0.UPF

K_POINTS gamma

"""


def primitiva(sup, rep):
    """Recorta una de las rep x rep celdas primitivas de la supercélula relajada."""
    f = sup.get_scaled_positions(wrap=True)[:, :2] * rep          # en unidades de la primitiva
    for o in np.linspace(0, 1, 50, endpoint=False):               # origen sin átomos en el borde
        d = (f - o) % rep
        if np.abs(d - np.round(d)).min() > 0.01:
            break
    prim = sup[np.all(d < 1, axis=1)]
    prim.set_cell([sup.cell[0] / rep, sup.cell[1] / rep, sup.cell[2]], scale_atoms=False)
    prim.wrap()
    if len(prim) * rep ** 2 != len(sup):
        sys.exit('%s no es una supercélula %dx%d de la primitiva' % (sys.argv[1], rep, rep))
    return prim


def escribe_input(ruta, a):
    with open(ruta, 'w') as fh:
        fh.write(QE.format(nat=len(a), conv_thr=CONV_THR_POR_ATOMO * len(a)))
        fh.write('CELL_PARAMETERS angstrom\n')
        fh.writelines('%16.10f %16.10f %16.10f\n' % tuple(v) for v in a.cell)
        fh.write('\nATOMIC_POSITIONS angstrom\n')
        fh.writelines('%-2s %16.10f %16.10f %16.10f\n' % (s, *p) for s, p in zip(a.get_chemical_symbols(), a.positions))


def main():
    sup = read(sys.argv[1] if len(sys.argv) > 1 else 'relax_final.xyz')
    prim = primitiva(sup, REP)
    print('Primitiva: %s, a = %.4f Å' % (prim.get_chemical_formula(), prim.cell.lengths()[0]))
    os.makedirs('laminas', exist_ok=True)
    nuevos = 0
    for ic, celda in enumerate(CELDAS):
        n1, n2 = map(int, celda.split('x'))
        lamina = prim.repeat((n1, n2, 1))
        lamina.pbc = (True, True, False)
        write('laminas/BGDY_%s.xyz' % celda, lamina)
        for js, s in enumerate(SIGMAS):
            for i in range(N_POR_SIGMA):
                d = 'casos/%s/s%.2f_%03d' % (celda, s, i)
                if os.path.exists(d + '/output.out'):
                    continue
                a = lamina.copy()
                a.rattle(stdev=s, seed=SEMILLA + 10000 * ic + 1000 * js + i)
                a.wrap()
                os.makedirs(d, exist_ok=True)
                escribe_input(d + '/input.in', a)
                nuevos += 1
        print('%-4s %4d átomos  %d casos' % (celda, len(lamina), len(SIGMAS) * N_POR_SIGMA))
    print('%d inputs escritos en casos/' % nuevos)


if __name__ == '__main__':
    main()
