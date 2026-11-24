#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convierte out.pw de Quantum ESPRESSO a XYZ extendido (OVITO)
Compatible con Python 2.7.5

Soporta ATOMIC_POSITIONS en:
- angstrom
- bohr
- alat
"""

import re
import sys

BOHR_TO_ANG = 0.52917721092


def parse_out_pw(filename):
    frames = []
    alat_ang = None
    cell = None

    f = open(filename, "r")
    lines = f.readlines()
    f.close()

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
                raise RuntimeError("ERROR: alat no definido")

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
                pos = map(float, parts[1:4])

                # ---- conversi�n correcta de unidades ----
                if "angstrom" in unit_line:
                    pos = list(pos)
                elif "bohr" in unit_line:
                    pos = [x * BOHR_TO_ANG for x in pos]
                elif "alat" in unit_line:
                    if alat_ang is None:
                        raise RuntimeError("ERROR: ATOMIC_POSITIONS (alat) sin alat")
                    pos = [x * alat_ang for x in pos]
                else:
                    raise RuntimeError(
                        "Unidad desconocida en ATOMIC_POSITIONS: " + line.strip()
                    )

                atoms.append((specie, pos))
                i += 1

            if cell is not None and atoms:
                frames.append((cell, atoms))
            continue

        i += 1

    return frames


def write_xyz(frames, output):
    f = open(output, "w")

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

    f.close()


def main():
    if len(sys.argv) != 3:
        print("Uso:")
        print("  python linux_qe_xyz.py out.pw movie.xyz")
        sys.exit(1)

    frames = parse_out_pw(sys.argv[1])

    if not frames:
        raise RuntimeError("No se encontraron frames")

    write_xyz(frames, sys.argv[2])

    print("OK: {} frames escritos en {}".format(len(frames), sys.argv[2]))


if __name__ == "__main__":
    main()
