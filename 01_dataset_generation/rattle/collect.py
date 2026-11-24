"""Junta los SCF terminados (cases/*/qe_out.out) en dataset.extxyz.  Uso: python collect.py   (necesita ase)"""
from pathlib import Path
from ase.io import read, write

frames, pendientes = [], []
for d in sorted(Path("cases").glob("case_*")):
    f = d / "qe_out.out"
    txt = f.read_text(errors="replace") if f.exists() else ""
    if "convergence has been achieved" not in txt or "JOB DONE" not in txt:
        pendientes.append(d.name)
        continue
    a = read(f, format="espresso-out")                       # energía (eV), fuerzas (eV/Å), tensión (eV/Å³)
    a.info["rattle_std"] = read(d / "initial.xyz").info["rattle_std"]
    a.info["virial"] = -a.get_volume() * a.get_stress(voigt=False)   # lámina 2D: usa el virial, no la tensión (QE la promedia con el vacío)
    frames.append(a)

write("dataset.extxyz", frames)
print(f"{len(frames)} casos en dataset.extxyz | sin terminar ({len(pendientes)}): {pendientes}")
