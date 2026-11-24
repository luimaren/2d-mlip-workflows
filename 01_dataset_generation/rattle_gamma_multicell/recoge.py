#!/usr/bin/env python3
"""Junta los scf terminados en dataset_rattle_gamma.extxyz y dice cuántos faltan por celda.

Claves: energy_qe (eV), forces_qe (eV/Å), stress_qe (eV/Å^3, signo ASE), como train_filtrado.xyz.
"""
import collections
import glob
import os
import re

from ase.io import read, write

datos, cuenta = [], collections.defaultdict(collections.Counter)
for d in sorted(glob.glob('casos/*/s*')):
    celda, caso = d.split(os.sep)[-2:]
    out = d + '/output.out'
    txt = open(out, errors='replace').read() if os.path.exists(out) else ''
    if 'JOB DONE' not in txt or 'convergence NOT achieved' in txt or txt.count('Program PWSCF') > 1:
        cuenta[celda]['sin terminar' if txt else 'sin output'] += 1
        continue
    a = read(out, format='espresso-out')
    sigma, idx = caso[1:].split('_')
    mag = re.findall(r'total magnetization\s+=\s+(-?[\d.]+)', txt)
    a.info = {'config_type': 'rattle_s' + sigma, 'celda': celda, 'sigma': float(sigma), 'idx': int(idx),
              'mag_tot': float(mag[-1]) if mag else 0.0,
              'energy_qe': a.get_potential_energy(), 'stress_qe': a.get_stress(voigt=False)}
    a.arrays['forces_qe'] = a.get_forces()
    a.calc = None
    a.pbc = (True, True, False)
    datos.append(a)
    cuenta[celda]['hecho'] += 1

write('dataset_rattle_gamma.extxyz', datos)
print('%-5s %6s %13s %11s' % ('celda', 'hecho', 'sin terminar', 'sin output'))
for celda, c in cuenta.items():
    print('%-5s %6d %13d %11d' % (celda, c['hecho'], c['sin terminar'], c['sin output']))
print('%d configuraciones -> dataset_rattle_gamma.extxyz' % len(datos))
