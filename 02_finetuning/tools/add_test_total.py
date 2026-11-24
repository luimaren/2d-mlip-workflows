#!/usr/bin/env python3
"""
Añade al xyz de test una copia de todas sus configuraciones con config_type=TOTAL.

MACE agrupa el test por config_type y saca una fila por grupo en "Error-table on TEST".
Con la copia, aparece además la fila "TOTAL_Default", calculada por el propio MACE sobre
todas las configuraciones de test (mismas métricas que train y valid).

- Las configuraciones originales se copian tal cual, byte a byte; la copia solo cambia
  el valor de config_type en la línea de comentario (o lo añade si no está).
- No necesita ASE ni MACE.
- El test no se usa para entrenar, así que el entrenamiento no cambia. La evaluación
  final del test tarda el doble (en v31: ~15 s).

Uso:
  python3 add_test_total.py test_filtrado.xyz test_filtrado_con_total.xyz
  python3 add_test_total.py test_filtrado.xyz salida.xyz --name TEST_TOTAL
"""
import argparse
import re
import sys

CT_RE = re.compile(r'(?<![\w-])config_type=("[^"]*"|\'[^\']*\'|\{[^}]*\}|\S+)')


def iter_frames(lines):
    """Devuelve (natoms, comentario, líneas_de_átomos) para cada frame de un xyz/extxyz."""
    i, n = 0, len(lines)
    k = 0
    while i < n:
        if not lines[i].strip():
            i += 1
            continue
        try:
            nat = int(lines[i].split()[0])
        except (ValueError, IndexError):
            sys.exit(f"Línea {i + 1}: esperaba el número de átomos del frame {k} y hay: {lines[i]!r}")
        if i + 2 + nat > n:
            sys.exit(f"Frame {k} (línea {i + 1}): el archivo se acaba antes de {nat} átomos")
        yield lines[i], lines[i + 1], lines[i + 2:i + 2 + nat]
        i += 2 + nat
        k += 1


def relabel(comment, name):
    if CT_RE.search(comment):
        return CT_RE.sub(f"config_type={name}", comment, count=1)
    return comment.rstrip("\r\n") + f" config_type={name}\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada")
    ap.add_argument("salida")
    ap.add_argument("--name", default="TOTAL", help="config_type del grupo total (fila '<name>_Default' en MACE)")
    args = ap.parse_args()

    with open(args.entrada, encoding="utf-8") as fh:
        lines = fh.readlines()
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    frames = list(iter_frames(lines))
    tipos = {}
    for _, com, _ in frames:
        m = CT_RE.search(com)
        ct = m.group(1).strip("\"'") if m else "Default"
        tipos[ct] = tipos.get(ct, 0) + 1
    if args.name in tipos:
        sys.exit(f"Ya hay configuraciones con config_type={args.name}: elige otro --name")

    with open(args.salida, "w", encoding="utf-8") as out:
        for head, com, atoms in frames:  # originales, sin tocar
            out.write(head)
            out.write(com)
            out.writelines(atoms)
        for head, com, atoms in frames:  # copia etiquetada como TOTAL
            out.write(head)
            out.write(relabel(com, args.name))
            out.writelines(atoms)

    print(f"{len(frames)} configuraciones en {len(tipos)} grupos + copia '{args.name}' "
          f"→ {args.salida} ({2 * len(frames)} frames)")
    for ct, c in tipos.items():
        print(f"  {ct}: {c}")
    print(f"  {args.name}: {len(frames)}")


if __name__ == "__main__":
    main()
