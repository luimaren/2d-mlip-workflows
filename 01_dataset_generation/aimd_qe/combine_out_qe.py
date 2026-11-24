#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
combine_qe_restarts_to_xyz.py

Convierte varios ficheros de salida (out.pw) de Quantum ESPRESSO -uno por
cada restart de la misma simulacion- a un UNICO fichero XYZ extendido
(formato OVITO/ASE), concatenando los frames de cada fichero EN EL ORDEN
en que los pases por linea de comandos.

Reutiliza la misma logica de parseo que linux_qe_xyz.py (lattice
parameter, crystal axes, ATOMIC_POSITIONS en angstrom/bohr/alat), asi que
soporta los mismos formatos de salida de pw.x.

------------------------------------------------------------------------
SOBRE LOS RESTARTS:
Cuando reinicias un calculo de QE (restart_mode='restart' o similar), es
habitual que el nuevo out.pw vuelva a imprimir como "primer frame" la
MISMA configuracion con la que termino el calculo anterior (el ultimo
paso convergido antes de cortarse). Si concatenas todo sin mas, esa
configuracion queda duplicada en el XYZ combinado.

Este script incluye un flag opcional --dedup que, si lo activas, elimina
el primer frame de cada fichero (excepto el primero de la lista) SI es
practicamente identico (misma celda y mismas posiciones, dentro de una
tolerancia) al ULTIMO frame del fichero anterior. Si no estas seguro,
puedes correrlo sin --dedup primero, mirar cuantos frames salen, y
decidir si hace falta.
------------------------------------------------------------------------

Uso:
    python3 combine_qe_restarts_to_xyz.py out1.pw out2.pw out3.pw \
        --output movie_combinado.xyz

    # con deduplicacion de frames repetidos en el arranque de cada restart:
    python3 combine_qe_restarts_to_xyz.py out1.pw out2.pw out3.pw \
        --output movie_combinado.xyz --dedup
"""

import argparse
import re
import sys

BOHR_TO_ANG = 0.52917721092


def parse_out_pw(filename):
    """Parsea un fichero de salida de pw.x y devuelve una lista de frames.

    Cada frame es una tupla (cell, atoms), donde:
      - cell es una lista 3x3 de floats (Angstrom)
      - atoms es una lista de tuplas (especie, [x, y, z]) en Angstrom
    """
    frames = []
    alat_ang = None
    cell = None

    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    nlines = len(lines)

    while i < nlines:
        line = lines[i]

        # ---------------------------------
        # lattice parameter (alat)
        # ---------------------------------
        if "lattice parameter (alat)" in line:
            alat_au = float(line.split("=")[1].split()[0])
            alat_ang = alat_au * BOHR_TO_ANG

        # ---------------------------------
        # crystal axes (units of alat)
        # ---------------------------------
        if "crystal axes:" in line:
            if alat_ang is None:
                raise RuntimeError(f"ERROR: alat no definido (fichero {filename})")

            axes = []
            for j in range(1, 4):
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", lines[i + j])
                axes.append([float(x) for x in nums[-3:]])

            cell = []
            for vec in axes:
                cell.append([alat_ang * x for x in vec])

            i += 4
            continue

        # ---------------------------------
        # ATOMIC_POSITIONS
        # ---------------------------------
        if "ATOMIC_POSITIONS" in line:
            atoms = []
            unit_line = line.lower()
            i += 1

            while i < nlines:
                if not lines[i].strip():
                    break
                parts = lines[i].split()
                if len(parts) < 4:
                    break

                specie = parts[0]
                pos = [float(x) for x in parts[1:4]]

                if "angstrom" in unit_line:
                    pass  # ya esta en Angstrom
                elif "bohr" in unit_line:
                    pos = [x * BOHR_TO_ANG for x in pos]
                elif "alat" in unit_line:
                    if alat_ang is None:
                        raise RuntimeError(
                            f"ERROR: ATOMIC_POSITIONS (alat) sin alat (fichero {filename})"
                        )
                    pos = [x * alat_ang for x in pos]
                else:
                    raise RuntimeError(
                        f"Unidad desconocida en ATOMIC_POSITIONS ({filename}): {line.strip()}"
                    )

                atoms.append((specie, pos))
                i += 1

            if cell is not None and atoms:
                frames.append((cell, atoms))
            continue

        i += 1

    return frames


def frames_are_equal(frame_a, frame_b, tol=1e-6):
    """Compara dos frames (cell, atoms) y dice si son practicamente
    identicos (misma celda y mismas posiciones dentro de una tolerancia).
    Se usa para detectar el frame duplicado que QE reimprime al reiniciar.
    """
    cell_a, atoms_a = frame_a
    cell_b, atoms_b = frame_b

    if len(atoms_a) != len(atoms_b):
        return False

    for row_a, row_b in zip(cell_a, cell_b):
        for va, vb in zip(row_a, row_b):
            if abs(va - vb) > tol:
                return False

    for (sp_a, pos_a), (sp_b, pos_b) in zip(atoms_a, atoms_b):
        if sp_a != sp_b:
            return False
        for va, vb in zip(pos_a, pos_b):
            if abs(va - vb) > tol:
                return False

    return True


def write_xyz(frames, output):
    with open(output, "w") as f:
        for cell, atoms in frames:
            f.write(str(len(atoms)) + "\n")

            lattice_vals = []
            for i in range(3):
                for j in range(3):
                    lattice_vals.append("{:.10f}".format(cell[i][j]))

            lattice = " ".join(lattice_vals)

            f.write(
                'Lattice="{}" Properties=species:S:1:pos:R:3\n'.format(lattice)
            )

            for specie, pos in atoms:
                f.write(
                    "{} {:.10f} {:.10f} {:.10f}\n".format(
                        specie, pos[0], pos[1], pos[2]
                    )
                )


def main():
    parser = argparse.ArgumentParser(
        description="Combina varios out.pw (restarts de QE) en un unico XYZ extendido."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Ficheros out.pw EN ORDEN (ej: out_part1.pw out_part2.pw out_part3.pw)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="combinado.xyz",
        help="Fichero XYZ combinado de salida (default: combinado.xyz)",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Elimina el primer frame de cada fichero (salvo el primero de la lista) "
        "si es identico al ultimo frame del fichero anterior (tipico al reiniciar).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Tolerancia (Angstrom) para considerar dos frames identicos con --dedup "
        "(default: 1e-6)",
    )
    args = parser.parse_args()

    all_frames = []
    per_file_counts = []

    for idx, fname in enumerate(args.inputs):
        frames = parse_out_pw(fname)
        if not frames:
            print(f"AVISO: no se encontraron frames en '{fname}'")

        if args.dedup and idx > 0 and all_frames and frames:
            if frames_are_equal(all_frames[-1], frames[0], tol=args.tol):
                print(
                    f"  -> dedup: se elimina el primer frame de '{fname}' "
                    f"(identico al ultimo de '{args.inputs[idx - 1]}')"
                )
                frames = frames[1:]

        per_file_counts.append((fname, len(frames)))
        all_frames.extend(frames)

    if not all_frames:
        raise RuntimeError("No se ha extraido ningun frame de ningun fichero de entrada.")

    write_xyz(all_frames, args.output)

    print("\nResumen:")
    for fname, n in per_file_counts:
        print(f"  {fname}: {n} frames")
    print(f"\nOK: {len(all_frames)} frames combinados escritos en '{args.output}'")


if __name__ == "__main__":
    main()
