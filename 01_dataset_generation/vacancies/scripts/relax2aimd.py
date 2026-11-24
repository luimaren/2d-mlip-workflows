#!/usr/bin/env python3
"""
relax2aimd.py  -  Del output de un relax de Quantum ESPRESSO a un input de AIMD.

Que hace
--------
1. Lee el output de pw.x (relax, vc-relax; tambien vale un scf) y extrae:
     - la estructura FINAL (ultimo bloque ATOMIC_POSITIONS del output),
     - toda la TRAYECTORIA de relajacion: la geometria inicial (seccion
       "Cartesian axes") mas cada ATOMIC_POSITIONS posterior, sin duplicados
       consecutivos (QE repite la geometria final al terminar),
     - la energia total de cada paso (lineas "!") para ponerla en el xyz.
2. En el input de AIMD BORRA las posiciones atomicas que haya (card
   ATOMIC_POSITIONS) y pone las de la ultima geometria relajada, con las
   especies y el orden del output. Actualiza nat en &SYSTEM con el numero de
   atomos del output (y CELL_PARAMETERS solo si hace falta). Todo lo demas
   (namelists, K_POINTS...) queda intacto. Las posiciones que hubiera en el
   input no se leen ni se comparan: solo se usa como plantilla.
   El input resultante NO relaja: solo lleva la estructura ya relajada.
3. Escribe dos ficheros .xyz (extended XYZ, con la celda y la energia en la
   linea de comentario de cada frame):
     - <input>_relax_final.xyz : la estructura final (1 frame)
     - <input>_relax_traj.xyz  : la trayectoria completa del relax

Uso
---
    python3 relax2aimd.py qe_out.out input_nvt.in
    python3 relax2aimd.py qe_out.out input_nvt.in -o aimd.in
    python3 relax2aimd.py qe_out.out input_nvt.in --inplace     # sobrescribe (deja .bak)

Solo usa la libreria estandar de Python (>= 3.6).
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

BOHR = 0.529177210903      # angstrom por bohr (CODATA 2018, el que usa QE 7.x)
RY_EV = 13.605693122994    # eV por Rydberg

FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"
ATOM_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s+(%s)\s+(%s)\s+(%s)" % (FLOAT, FLOAT, FLOAT))
TAU_RE = re.compile(
    r"^\s*\d+\s+([A-Za-z][A-Za-z0-9_]*)\s+tau\(\s*\d+\)\s*=\s*\(\s*(%s)\s+(%s)\s+(%s)\s*\)"
    % (FLOAT, FLOAT, FLOAT)
)
VEC_RE = re.compile(r"^\s*(%s)\s+(%s)\s+(%s)\s*$" % (FLOAT, FLOAT, FLOAT))
ENERGY_RE = re.compile(r"^!\s+total energy\s+=\s+(%s)\s+Ry" % FLOAT)
BFGS_OK_RE = re.compile(r"bfgs converged in\s+(\d+)\s+scf cycles and\s+(\d+)\s+bfgs steps")
CARD_RE = re.compile(
    r"^\s*(ATOMIC_SPECIES|ATOMIC_POSITIONS|K_POINTS|CELL_PARAMETERS|OCCUPATIONS|"
    r"CONSTRAINTS|ATOMIC_FORCES|ADDITIONAL_K_POINTS|HUBBARD|ATOMIC_VELOCITIES)\b",
    re.I,
)


def num(s):
    return float(s.replace("D", "E").replace("d", "e"))


def die(msg):
    sys.exit("ERROR: " + msg)


# ----------------------------------------------------------------------------
# Unidades
# ----------------------------------------------------------------------------
def parse_unit(header):
    """(unidad, valor_alat_o_None) a partir de la cabecera de una card."""
    m = re.search(r"alat\s*=\s*(%s)" % FLOAT, header, re.I)
    if m:
        return "alat", num(m.group(1))
    h = header.lower()
    for u in ("angstrom", "bohr", "crystal", "alat"):
        if u in h:
            return u, None
    return None, None


def to_angstrom(vecs, unit, alat_val, alat_bohr, cell=None):
    """Convierte una lista de vectores [x, y, z] a angstrom."""
    if unit == "angstrom":
        return [list(v) for v in vecs]
    if unit == "bohr":
        return [[c * BOHR for c in v] for v in vecs]
    if unit == "alat":
        a = (alat_val if alat_val else alat_bohr) * BOHR
        return [[c * a for c in v] for v in vecs]
    if unit == "crystal":
        if cell is None:
            die("Posiciones en 'crystal' pero no hay celda para convertirlas.")
        return [[sum(f[i] * cell[i][k] for i in range(3)) for k in range(3)] for f in vecs]
    die("Unidad no reconocida: %r" % (unit,))


def max_abs_diff(a, b):
    return max(abs(x - y) for p, q in zip(a, b) for x, y in zip(p, q))


# ----------------------------------------------------------------------------
# Lectura del output de QE
# ----------------------------------------------------------------------------
def parse_output(path):
    lines = Path(path).read_text(errors="replace").splitlines()

    nat = None
    for ln in lines:
        m = re.search(r"number of atoms/cell\s*=\s*(\d+)", ln)
        if m:
            nat = int(m.group(1))
            break
    if nat is None:
        die("No encuentro 'number of atoms/cell' en %s. Es un output de pw.x?" % path)

    # alat en bohr (celldm(1) trae 6 decimales; 'lattice parameter' solo 4)
    alat_bohr = None
    for ln in lines:
        m = re.search(r"celldm\(1\)\s*=\s*(%s)" % FLOAT, ln)
        if m:
            alat_bohr = num(m.group(1))
            break
    if alat_bohr is None:
        for ln in lines:
            m = re.search(r"lattice parameter \(alat\)\s*=\s*(%s)" % FLOAT, ln)
            if m:
                alat_bohr = num(m.group(1))
                break
    if alat_bohr is None:
        die("No encuentro el parametro de red (alat) en el output.")

    # celda inicial: crystal axes
    cell0 = None
    for i, ln in enumerate(lines):
        if "crystal axes" in ln and "alat" in ln:
            rows = []
            for l2 in lines[i + 1:i + 4]:
                m = re.search(r"a\(\d\)\s*=\s*\(\s*(%s)\s+(%s)\s+(%s)\s*\)" % (FLOAT, FLOAT, FLOAT), l2)
                if m:
                    rows.append([num(m.group(k)) for k in (1, 2, 3)])
            if len(rows) == 3:
                cell0 = to_angstrom(rows, "alat", alat_bohr, alat_bohr)
            break

    # geometria inicial: posiciones tau (unidades de alat) impresas al principio
    species0, pos0 = None, None
    for i, ln in enumerate(lines):
        if "positions (alat units)" in ln:
            sp, raw = [], []
            for l2 in lines[i + 1:]:
                m = TAU_RE.match(l2)
                if not m:
                    break
                sp.append(m.group(1))
                raw.append([num(m.group(k)) for k in (2, 3, 4)])
            if len(sp) == nat:
                species0 = sp
                pos0 = to_angstrom(raw, "alat", None, alat_bohr)
            break

    # energias totales de cada ciclo SCF (Ry -> eV)
    energies = []
    for ln in lines:
        if ln.startswith("!"):
            m = ENERGY_RE.match(ln)
            if m:
                energies.append(num(m.group(1)) * RY_EV)

    # convergencia del relax
    conv = None
    for ln in lines:
        m = BFGS_OK_RE.search(ln)
        if m:
            conv = (int(m.group(1)), int(m.group(2)))

    # barrido secuencial: CELL_PARAMETERS y ATOMIC_POSITIONS en orden de aparicion
    frames = []
    cell = cell0
    has_initial = pos0 is not None and cell0 is not None
    if has_initial:
        frames.append({"species": species0, "pos": pos0, "cell": cell0})
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].lstrip()
        if s.startswith("CELL_PARAMETERS"):
            unit, av = parse_unit(lines[i])
            rows = []
            for l2 in lines[i + 1:i + 4]:
                m = VEC_RE.match(l2)
                if not m:
                    break
                rows.append([num(m.group(k)) for k in (1, 2, 3)])
            if len(rows) == 3:
                cell = to_angstrom(rows, unit, av, alat_bohr)
                i += 4
                continue
        elif s.startswith("ATOMIC_POSITIONS"):
            unit, av = parse_unit(lines[i])
            sp, raw = [], []
            j = i + 1
            while j < n:
                m = ATOM_RE.match(lines[j])
                if not m:
                    break
                sp.append(m.group(1))
                raw.append([num(m.group(k)) for k in (2, 3, 4)])
                j += 1
            if len(sp) == nat:
                if cell is None:
                    die("Hay posiciones pero no he podido leer la celda del output.")
                if unit is None:
                    die("ATOMIC_POSITIONS sin unidad reconocible en el output.")
                frames.append({"species": sp, "pos": to_angstrom(raw, unit, av, alat_bohr, cell),
                               "cell": cell})
            i = j
            continue
        i += 1

    if not frames:
        die("No he encontrado posiciones atomicas completas en el output.")

    # energia de cada frame: el frame j (0 = inicial) se calculo en el ciclo SCF j+1
    off = 0 if has_initial else 1
    for j, fr in enumerate(frames):
        k = j + off
        fr["energy"] = energies[k] if k < len(energies) else None

    # quitar duplicados consecutivos (QE repite la geometria final)
    unique = [frames[0]]
    for fr in frames[1:]:
        prev = unique[-1]
        if max_abs_diff(fr["pos"], prev["pos"]) < 1e-8 and max_abs_diff(fr["cell"], prev["cell"]) < 1e-8:
            continue
        unique.append(fr)

    return dict(nat=nat, alat_bohr=alat_bohr, frames=unique, n_raw=len(frames),
                has_initial=has_initial, conv=conv, n_energies=len(energies))


# ----------------------------------------------------------------------------
# Lectura / edicion del input
# ----------------------------------------------------------------------------
def card_bounds(lines, name):
    """(inicio, fin) de una card; fin excluye lineas en blanco finales. None si no existe."""
    cards = [(i, CARD_RE.match(ln).group(1).upper()) for i, ln in enumerate(lines) if CARD_RE.match(ln)]
    for k, (i, nm) in enumerate(cards):
        if nm == name:
            end = cards[k + 1][0] if k + 1 < len(cards) else len(lines)
            while end > i + 1 and lines[end - 1].strip() == "":
                end -= 1
            return i, end
    return None


def parse_input_cell(lines, alat_bohr):
    """Celda del input en angstrom (None si no hay CELL_PARAMETERS). Las posiciones no se leen."""
    if card_bounds(lines, "ATOMIC_POSITIONS") is None:
        die("El input no tiene card ATOMIC_POSITIONS (no se donde poner las posiciones).")
    bc = card_bounds(lines, "CELL_PARAMETERS")
    if not bc:
        return None
    cu, cav = parse_unit(lines[bc[0]])
    rows = [[num(m.group(k)) for k in (1, 2, 3)]
            for m in (VEC_RE.match(l) for l in lines[bc[0] + 1:bc[1]]) if m]
    if len(rows) != 3:
        return None
    return to_angstrom(rows, cu or "angstrom", cav, alat_bohr)


def replace_card(lines, name, new_block):
    b = card_bounds(lines, name)
    if b is None:
        return lines
    return lines[:b[0]] + new_block + lines[b[1]:]


def species_labels(lines):
    """Etiquetas de ATOMIC_SPECIES del input (vacio si no hay card)."""
    b = card_bounds(lines, "ATOMIC_SPECIES")
    if b is None:
        return []
    labels = []
    for ln in lines[b[0] + 1:b[1]]:
        t = ln.split()
        if t and not t[0].startswith(("!", "#")):
            labels.append(t[0])
    return labels


def set_nat(lines, nat):
    """Pone nat=<nat> dentro de &SYSTEM. Devuelve (lineas, valor_anterior|None)."""
    nat_re = re.compile(r"(?<![A-Za-z0-9_])(nat\s*=\s*)(\d+)", re.I)
    in_sys, old, out = False, None, []
    for ln in lines:
        st = ln.strip().lower()
        if st.startswith("&system"):
            in_sys = True
        elif in_sys and (st == "/" or st.startswith("&")):
            in_sys = False
        if in_sys and old is None:
            m = nat_re.search(ln)
            if m:
                old = int(m.group(2))
                ln = nat_re.sub(lambda mm: "%s%d" % (mm.group(1), nat), ln, count=1)
        out.append(ln)
    return out, old


def fmt_positions(species, pos):
    out = ["ATOMIC_POSITIONS angstrom"]
    for s, (x, y, z) in zip(species, pos):
        out.append("%-3s %16.10f %16.10f %16.10f" % (s, x, y, z))
    return out


def fmt_cell(cell):
    out = ["CELL_PARAMETERS angstrom"]
    for r in cell:
        out.append("   %.10f %.10f %.10f" % (r[0], r[1], r[2]))
    return out


# ----------------------------------------------------------------------------
# Escritura de xyz (extended XYZ)
# ----------------------------------------------------------------------------
def xyz_frame(species, pos, cell, step=None, energy=None):
    lat = " ".join("%.10f" % v for row in cell for v in row)
    extra = ""
    if energy is not None:
        extra += " energy=%.8f" % energy        # eV, como espera ASE
    if step is not None:
        extra += " step=%d" % step
    head = '%d\nLattice="%s" Properties=species:S:1:pos:R:3 pbc="T T T"%s\n' % (len(species), lat, extra)
    body = "".join("%-3s %16.10f %16.10f %16.10f\n" % (s, x, y, z) for s, (x, y, z) in zip(species, pos))
    return head + body


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Estructura final y trayectoria de un relax de QE -> input de AIMD + 2 xyz")
    ap.add_argument("qe_out", help="output de pw.x (relax o vc-relax)")
    ap.add_argument("qe_in", help="input de AIMD cuya estructura se sustituye")
    ap.add_argument("-o", "--output", help="input resultante (def.: <input>_relax.in)")
    ap.add_argument("--xyz-final", help="xyz de la estructura final (def.: <input>_relax_final.xyz)")
    ap.add_argument("--xyz-traj", help="xyz de la trayectoria (def.: <input>_relax_traj.xyz)")
    ap.add_argument("--inplace", action="store_true",
                    help="sobrescribe el input original (guarda copia .bak)")
    ap.add_argument("--cell", choices=["auto", "output", "input"], default="auto",
                    help="origen de la celda: auto (def.) conserva la del input si coincide con la "
                         "del output (1e-3 A), 'output' siempre la del output, 'input' nunca la toca")
    args = ap.parse_args()

    for p in (args.qe_out, args.qe_in):
        if not Path(p).is_file():
            die("no puedo leer el fichero %s" % p)

    qin = Path(args.qe_in)
    out = parse_output(args.qe_out)
    frames = out["frames"]
    first, final = frames[0], frames[-1]

    lines = qin.read_text().splitlines()
    cell_in = parse_input_cell(lines, out["alat_bohr"])

    warnings = []
    labels = species_labels(lines)
    missing = sorted(set(final["species"]) - set(labels)) if labels else []
    if missing:
        warnings.append("Especies del output que NO estan en ATOMIC_SPECIES del input: %s "
                        "(anadelas a mano y ajusta ntyp)." % ", ".join(missing))

    # --- decision sobre la celda ---
    cell_final = final["cell"]
    replace_cell = False
    if cell_in is None:
        cell_msg = "el input no tiene CELL_PARAMETERS (ibrav != 0?): no se toca"
    else:
        dmax = max_abs_diff(cell_in, final["cell"])
        if args.cell == "input" or (args.cell == "auto" and dmax < 1e-3):
            cell_final = cell_in
            cell_msg = "se conserva la del input (difiere %.1e A de la del output)" % dmax
        else:
            replace_cell = True
            cell_msg = "sustituida por la del output (diferia %.1e A)" % dmax

    # --- nuevo input ---
    new = replace_card(lines, "ATOMIC_POSITIONS", fmt_positions(final["species"], final["pos"]))
    if replace_cell:
        new = replace_card(new, "CELL_PARAMETERS", fmt_cell(cell_final))
    new, nat_old = set_nat(new, out["nat"])
    if nat_old is None:
        warnings.append("No he encontrado 'nat' en &SYSTEM del input: revisalo a mano (debe ser %d)." % out["nat"])

    if args.inplace:
        shutil.copy(str(qin), str(qin) + ".bak")
        out_in = qin
    else:
        out_in = Path(args.output) if args.output else qin.with_name(qin.stem + "_relax" + qin.suffix)
    out_in.write_text("\n".join(new) + "\n")

    base = out_in.with_suffix("")
    xyz_final = Path(args.xyz_final) if args.xyz_final else Path(str(base) + "_final.xyz")
    xyz_traj = Path(args.xyz_traj) if args.xyz_traj else Path(str(base) + "_traj.xyz")
    if args.inplace and not args.xyz_final:
        xyz_final = qin.with_name(qin.stem + "_relax_final.xyz")
    if args.inplace and not args.xyz_traj:
        xyz_traj = qin.with_name(qin.stem + "_relax_traj.xyz")

    # --- xyz final ---
    n_last = len(frames) - 1
    xyz_final.write_text(xyz_frame(final["species"], final["pos"], cell_final,
                                   step=n_last, energy=final["energy"]))

    # --- xyz trayectoria (celda fija si es la misma en todos los frames) ---
    fixed = all(max_abs_diff(fr["cell"], final["cell"]) < 1e-6 for fr in frames)
    with open(str(xyz_traj), "w") as fh:
        for j, fr in enumerate(frames):
            fh.write(xyz_frame(fr["species"], fr["pos"], cell_final if fixed else fr["cell"],
                               step=j, energy=fr["energy"]))

    # --- resumen ---
    comp = ", ".join("%s=%d" % (e, final["species"].count(e)) for e in sorted(set(final["species"])))
    print("Atomos            : %d  (%s)" % (out["nat"], comp))
    if out["conv"]:
        print("Relax             : convergido (%d pasos BFGS, %d ciclos SCF)" % (out["conv"][1], out["conv"][0]))
    else:
        print("ATENCION          : el output NO indica 'bfgs converged'; el relax puede estar incompleto.\n"
              "                    Se usa la ultima geometria disponible.")
    print("Trayectoria       : %d frames unicos (%d bloques leidos, %d duplicados descartados)"
          % (len(frames), out["n_raw"], out["n_raw"] - len(frames)))
    if first["energy"] is not None and final["energy"] is not None:
        print("Energia           : %.6f eV (inicial) -> %.6f eV (final);  dE = %.6f eV"
              % (first["energy"], final["energy"], final["energy"] - first["energy"]))
    print("nat en el input   : %s" % ("%d -> %d" % (nat_old, out["nat"]) if nat_old is not None and nat_old != out["nat"]
                                      else "%d (sin cambios)" % out["nat"]))
    print("Celda             : %s" % cell_msg)
    print("Input escrito en  : %s%s" % (out_in, "  (original en .bak)" if args.inplace else ""))
    print("XYZ final         : %s" % xyz_final)
    print("XYZ trayectoria   : %s" % xyz_traj)
    if warnings:
        print("\nAVISOS:")
        for w in warnings:
            print("  " + w if not w.startswith(" ") else w)


if __name__ == "__main__":
    main()
