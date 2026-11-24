#!/usr/bin/env python3
"""Flujo strain -> scf -> dataset. Lo llaman 02_submit_scf.sh, 03_status.sh y 04_collect.sh:
    python workflow.py {submit,status,collect} [--cases ...]
--cases: numeros (3), rangos (5-8), listas (1,3,9) o trozos del nombre (uni-x, biaxial, aniso, m15). Sin --cases: todos.
Modos: uni-x, uni-y, biaxial (exx = eyy) y aniso (biaxial anisotropo: exx y eyy independientes, opcional exy)."""
import argparse
import csv
import getpass
import re
import signal
import subprocess
import sys
from collections import Counter
from pathlib import Path

if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)                    # "| head" sin traceback

ROOT = Path(__file__).resolve().parent
RY, BOHR, GPA = 13.605693122994, 0.529177210903, 160.21766208      # eV/Ry, A/bohr, GPa por eV/A^3
GROUPS = {"uniaxial_x": ("uni-x",), "uniaxial_y": ("uni-y",), "uniaxial": ("uni-x", "uni-y"),
          "biaxial": ("biaxial",), "aniso": ("aniso",), "total": ("uni-x", "uni-y", "biaxial", "aniso")}
RESEND = ("OK", "ERROR QE", "NO CONVERGE", "INTERRUMPIDO", "SIN SALIDA")      # estados que --force reenvia


# ---------------------------------------------------------------- casos y estado
def num(d):
    return int(d.name.split("_")[1])


def select_cases(tokens):
    """Carpetas de cases/ elegidas con --cases (todas si no se da)."""
    dirs = sorted((d for d in (ROOT / "cases").glob("case_*") if re.match(r"case_\d+_", d.name)), key=num)
    if not dirs:
        sys.exit("no hay casos: genera cases/ con 01_make_strain.ipynb")
    if not tokens:
        return dirs
    sel = []
    for part in ",".join(tokens).split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        hit = [d for d in dirs if int(m[1]) <= num(d) <= int(m[2] or m[1])] if m else [d for d in dirs if part in d.name]
        if not hit:
            print(f"aviso: '{part}' no coincide con ningun caso")
        sel += [d for d in hit if d not in sel]
    if not sel:
        sys.exit("--cases no selecciona ningun caso")
    return sorted(sel, key=num)


def queue(strict=True):
    """{jobid: estado} de tus jobs en Slurm."""
    try:
        out = subprocess.check_output(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i %T"], universal_newlines=True)
    except (OSError, subprocess.CalledProcessError) as ex:
        if strict:
            sys.exit(f"no puedo leer la cola ({ex}): lanza esto desde el cluster")
        return {}
    return dict(re.findall(r"^(\S+)\s+(\S+)$", out, re.M))


def secs(s):
    """'16473.66s' | '3m51.12s' | '3h57m' -> segundos."""
    return sum(float(v) * {"d": 86400, "h": 3600, "m": 60, "s": 1}[u] for v, u in re.findall(r"([\d.]+)\s*([dhms])", s))


def why(err):
    """Motivo de una interrupcion, segun el .err de Slurm."""
    t = err.read_text(errors="replace") if err.exists() else ""
    for key, msg in (("TIME LIMIT", "limite de tiempo"), ("oom_kill", "sin memoria (OOM)"), ("CANCELLED", "cancelado")):
        if key in t:
            return msg
    return ([l.strip() for l in t.splitlines() if "error" in l.lower()] or [""])[-1][:100]


def state(d, q):
    """(estado, detalle) de un caso: cola de Slurm (q) + qe_out.out."""
    jid = (d / ".jobid").read_text().split(";")[0].strip() if (d / ".jobid").exists() else ""
    txt = (d / "qe_out.out").read_text(errors="replace") if (d / "qe_out.out").exists() else None
    if q.get(jid) in ("PENDING", "CONFIGURING"):
        return "EN COLA", ""
    if q.get(jid) in ("RUNNING", "COMPLETING", "SUSPENDED"):
        it = re.findall(r"iteration #\s*(\d+)", txt or "")
        return "EJECUTANDO", f"iteracion {it[-1]}" if it else ""
    if txt is None:
        return ("SIN SALIDA", "cancelado?") if jid else ("SIN ENVIAR", "")
    if "Error in routine" in txt:
        m = re.search(r"Error in routine\s+(.*(?:\n.*)?)", txt)
        return "ERROR QE", " ".join(m.group(1).split()) if m else ""
    if "convergence NOT achieved" in txt:
        return "NO CONVERGE", ""
    if "JOB DONE" in txt and "convergence has been achieved" in txt:
        n = re.findall(r"has been achieved in\s+(\d+)", txt)[-1]
        w = re.search(r"PWSCF\s*:.*?CPU\s*(.*?)\s*WALL", txt)
        return "OK", f"{n} it" + (f", {secs(w.group(1)) / 60:.1f} min" if w else "")
    return "INTERRUMPIDO", why(d / "qe_out.err")


def label(s, det):
    return s + (f" ({det})" if det else "")


# ---------------------------------------------------------------- submit
def cmd_submit(args):
    q, sent, dirs = queue(), 0, select_cases(args.cases)
    w = max(24, *(len(d.name) for d in dirs))                        # los nombres aniso son mas largos
    for d in dirs:
        s, det = state(d, q)
        print(f"{d.name:{w}s} {label(s, det)}")
        if s != "SIN ENVIAR" and not (args.force and s in RESEND):
            print("   salto (" + {"EN COLA": "ya esta en la cola", "EJECUTANDO": "ya se esta ejecutando"}.get(s, "usa --force para reenviarlo") + ")")
            continue
        if args.max and sent >= args.max:
            print(f"   parado: --max {args.max} alcanzado")
            break
        if args.dry_run:
            print("   [dry-run] sbatch job.sh")
            sent += 1
            continue
        if s != "SIN ENVIAR":                                        # reenvio: aparta la salida anterior
            k = len(list(d.glob("qe_out.try*.out"))) + 1
            for ext in ("out", "err"):
                if (d / f"qe_out.{ext}").exists():
                    (d / f"qe_out.{ext}").rename(d / f"qe_out.try{k}.{ext}")
            print(f"   salida anterior -> qe_out.try{k}.out/.err")
        try:
            r = subprocess.run(["sbatch", "--parsable", f"--job-name={d.name}", str(ROOT / "job.sh")], cwd=str(d),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        except OSError:
            sys.exit("no encuentro sbatch: lanza esto desde el cluster")
        if r.returncode:
            print("   FALLO sbatch:", r.stderr.strip())
            continue
        (d / ".jobid").write_text(r.stdout.strip() + "\n")
        print(f"   enviado: job {r.stdout.strip()}")
        sent += 1
    print(f"\n{sent} trabajo(s) {'simulados (dry-run)' if args.dry_run else 'enviados'}.")


# ---------------------------------------------------------------- status
def cmd_status(args):
    q, count, dirs = queue(), Counter(), select_cases(args.cases)
    e = {d: dict(re.findall(r"(strain_\w+)=(\S+)", (d / "case.xyz").read_text().splitlines()[1])) for d in dirs}
    w = max(24, *(len(d.name) for d in dirs))                        # los nombres aniso son mas largos
    xy = any(float(v.get("strain_xy_pct", 0)) for v in e.values())   # columna e_xy solo si hay cizalla
    print(f"{'caso':{w}s} {'e_xx(%)':>8s} {'e_yy(%)':>8s}" + (f" {'e_xy(%)':>8s}" if xy else "") + "  estado")
    for d in dirs:
        s, det = state(d, q)
        count[s] += 1
        x, y = (float(e[d].get(k, "nan")) for k in ("strain_xx_pct", "strain_yy_pct"))
        z = float(e[d].get("strain_xy_pct", 0))
        print(f"{d.name:{w}s} {x:+8.2f} {y:+8.2f}" + (f" {z:+8.2f}" if xy else "") + f"  {label(s, det)}")
    print(f"\nResumen ({len(dirs)} casos): " + " | ".join(f"{k} {v}" for k, v in sorted(count.items())))


# ---------------------------------------------------------------- collect
def parse(txt, nat):
    """Energia (eV), fuerzas (eV/A), tension 3x3 (eV/A^3, signo ASE: traccion > 0), posiciones (A), magnetizacion, iteraciones."""
    import numpy as np

    def last(pattern):                                               # ultimo valor numerico (nan si no hay)
        m = re.findall(pattern, txt, re.M)
        return float(m[-1]) if m else float("nan")

    assert "Forces acting" in txt and "total   stress" in txt, "faltan fuerzas o tension"
    E = last(r"^!\s+total energy\s+=\s+(\S+)") * RY                  # energia libre de QE
    F = np.array(re.findall(r"force =\s+(\S+)\s+(\S+)\s+(\S+)",
                            txt.rsplit("Forces acting", 1)[1].split("Total force")[0]), float) * RY / BOHR
    S = -np.array([l.split()[:3] for l in txt.rsplit("total   stress", 1)[1].splitlines()[1:4]], float) * RY / BOHR ** 3
    alat = last(r"celldm\(1\)=\s*(\S+)") * BOHR
    pos = np.array(re.findall(r"tau\(\s*\d+\) = \(\s*(\S+)\s+(\S+)\s+(\S+)", txt)[:nat], float) * alat
    assert np.isfinite(E) and F.shape == (nat, 3) and S.shape == (3, 3), "output incompleto"
    return E, F, S, pos, last(r"total magnetization\s+=\s+(\S+)"), int(last(r"convergence has been achieved in\s+(\d+)"))


def cmd_collect(args):
    import numpy as np                                               # numpy y ASE solo hacen falta aqui
    from ase.io import read, write
    q, dirs, frames, bad = queue(False), select_cases(args.cases), [], []
    for d in dirs:
        s, det = state(d, q)
        if s != "OK":
            bad.append((d.name, label(s, det)))
            continue
        try:
            a = read(d / "case.xyz")
            E, F, S, pos, mag, it = parse((d / "qe_out.out").read_text(errors="replace"), len(a))
            assert pos.shape == a.positions.shape and np.abs(pos - a.positions).max() < 1e-3, "el output no corresponde a case.xyz"
        except Exception as ex:
            bad.append((d.name, f"no se pudo leer: {ex}"))
            continue
        a.info.update(REF_energy=E, REF_stress=S, magnetization=mag, scf_iters=it, config_type="strain_" + a.info["strain_mode"])
        a.info.setdefault("strain_xy_pct", 0.0)                       # mismas claves en todos los modos
        a.info.setdefault("rattle_A", 0.0)
        a.arrays["REF_forces"] = F
        frames.append(a)

    if not frames:
        print("ningun caso OK: no se escribe nada")
    else:
        out = ROOT / "dataset"
        out.mkdir(exist_ok=True)
        print("grupo        casos")
        for g, modes in GROUPS.items():
            sel = [a for a in frames if a.info["strain_mode"] in modes]
            if sel:
                write(out / f"{g}.extxyz", sel)
            elif (out / f"{g}.extxyz").exists():
                (out / f"{g}.extxyz").unlink()                       # no dejar un fichero de otra tanda
            print(f"{g:12s} {len(sel)}/{sum(d.name.split('_')[2] in modes for d in dirs)}")
        fuera = Counter(a.info["strain_mode"] for a in frames if not any(a.info["strain_mode"] in m for m in GROUPS.values()))
        for m, n in fuera.items():
            print(f"aviso: {n} caso(s) del modo '{m}' no entran en ningun grupo (anadelo a GROUPS)")
        with open(str(out / "summary.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case", "mode", "e_xx_pct", "e_yy_pct", "e_xy_pct", "energy_eV", "fmax_eV_A", "sxx_GPa", "syy_GPa", "sxy_GPa",
                        "magnetization", "scf_iters"])
            for a in frames:
                i, s = a.info, a.info["REF_stress"] * GPA
                w.writerow([i["case"], i["strain_mode"], i["strain_xx_pct"], i["strain_yy_pct"], i["strain_xy_pct"], f"{i['REF_energy']:.6f}",
                            f"{np.abs(a.arrays['REF_forces']).max():.5f}", f"{s[0, 0]:.4f}", f"{s[1, 1]:.4f}", f"{s[0, 1]:.4f}",
                            i["magnetization"], i["scf_iters"]])
    for name, why_ in bad:
        print(f"  {name}: {why_}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.required = True
    for name in ("submit", "status", "collect"):
        sub.add_parser(name).add_argument("--cases", nargs="+", help="numeros, rangos (5-8), listas (1,3,9) o trozos del nombre")
    p = sub.choices["submit"]
    p.add_argument("--max", type=int, default=0, help="envia como mucho N trabajos")
    p.add_argument("--dry-run", action="store_true", help="no envia nada, solo muestra")
    p.add_argument("--force", action="store_true", help="reenvia tambien los terminados, con error o interrumpidos")
    args = ap.parse_args()
    {"submit": cmd_submit, "status": cmd_status, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
