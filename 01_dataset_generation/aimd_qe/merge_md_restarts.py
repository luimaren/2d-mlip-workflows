#!/usr/bin/env python3
"""
merge_md_restarts.py

Fusiona los distintos ficheros de salida de una dinamica QE que tuvo que
reiniciarse varias veces (output.1, output.2, output.3, ...) en una unica
trayectoria, y la escribe como un solo .xyz extendido (formato usado por
MACE para entrenamiento).

Cada output.N puede contener varios pasos ionicos (es una salida de
'md'/'vc-relax', no un scf puntual), asi que se leen TODOS los pasos de
cada fichero con ASE, no solo el ultimo.

Al reiniciar, QE normalmente vuelve a escribir como primer paso el ultimo
paso ya guardado en el fichero anterior (el que se uso para reiniciar), por
lo que ese paso quedaria duplicado en la trayectoria fusionada. El script
detecta y elimina ese solape comparando energia (y opcionalmente numero de
atomos/posiciones) entre el ultimo frame de un fichero y el primero del
siguiente.

Uso:
    python merge_md_restarts.py [carpeta_dinamica] [salida.xyz]

Si no se pasan argumentos, usa la carpeta actual y escribe trajectory.xyz
"""

import os
import re
import sys
from ase.io import read, write

# tolerancia (eV) para considerar que dos frames son "el mismo paso"
ENERGY_TOL = 1e-6


def encontrar_ficheros_restart(carpeta, patron=r"^output\.(\d+)$"):
    """Busca output.1, output.2, ... en la carpeta y los devuelve ordenados
    numericamente por el indice de reinicio."""
    regex = re.compile(patron)
    encontrados = []
    for f in os.listdir(carpeta):
        m = regex.match(f)
        if m:
            encontrados.append((int(m.group(1)), f))
    encontrados.sort(key=lambda x: x[0])
    return [os.path.join(carpeta, f) for _, f in encontrados]


def leer_pasos(out_file):
    """Lee todos los pasos ionicos de un output de QE (md/vc-relax) con ASE.
    Devuelve lista de Atoms (puede estar vacia si no hay pasos validos)."""
    try:
        frames = read(out_file, format="espresso-out", index=":")
    except Exception as e:
        print(f"  [aviso] no se pudo parsear {out_file}: {e}")
        return []

    # read(..., index=':') puede devolver un solo Atoms si solo hay 1 paso
    if not isinstance(frames, list):
        frames = [frames]

    # descarta frames sin energia calculada
    frames_validos = []
    for atoms in frames:
        if atoms.calc is not None and atoms.calc.results.get("energy") is not None:
            frames_validos.append(atoms)
    return frames_validos


def mismo_paso(a, b):
    """Compara dos Atoms para saber si son el mismo paso ionico (solape de
    reinicio): misma energia (dentro de tolerancia) y mismo numero de
    atomos."""
    if len(a) != len(b):
        return False
    ea = a.calc.results.get("energy")
    eb = b.calc.results.get("energy")
    if ea is None or eb is None:
        return False
    return abs(ea - eb) < ENERGY_TOL


def fusionar_dinamica(carpeta):
    """Devuelve la lista de Atoms de la trayectoria completa de una carpeta,
    fusionando sus output.N y quitando el solape entre reinicios."""
    ficheros = encontrar_ficheros_restart(carpeta)

    if not ficheros:
        print(f"  [aviso] no se encontraron ficheros output.N en {carpeta}")
        return []

    print(f"  ficheros detectados (en orden): {[os.path.basename(f) for f in ficheros]}")

    trayectoria = []
    for out_file in ficheros:
        pasos = leer_pasos(out_file)
        if not pasos:
            print(f"  [aviso] {os.path.basename(out_file)} sin pasos validos, se ignora")
            continue

        if trayectoria and mismo_paso(trayectoria[-1], pasos[0]):
            # el primer paso de este fichero es el ultimo del anterior: se descarta
            pasos = pasos[1:]

        print(f"  {os.path.basename(out_file)}: {len(pasos)} pasos añadidos")
        trayectoria.extend(pasos)

    return trayectoria


def main():
    carpeta = sys.argv[1] if len(sys.argv) > 1 else "."
    salida = sys.argv[2] if len(sys.argv) > 2 else "trajectory.xyz"

    print(f"Procesando dinamica en: {carpeta}")
    trayectoria = fusionar_dinamica(carpeta)

    if not trayectoria:
        print("No se genero ningun frame valido, no se escribe fichero.")
        return

    write(salida, trayectoria, format="extxyz")
    print(f"\nTrayectoria fusionada: {len(trayectoria)} pasos")
    print(f"Escrita en: {salida}")


if __name__ == "__main__":
    main()
