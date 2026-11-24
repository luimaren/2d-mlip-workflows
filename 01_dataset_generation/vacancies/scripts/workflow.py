#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
workflow.py -- vacantes (xyz) -> relax QE -> relax2aimd -> NVT QE, en un cluster con Slurm.

Solo libreria estandar (no necesita ASE ni numpy). Se usa a traves de los wrappers 01_..06_
de la raiz del proyecto, o directamente:

    python3 scripts/workflow.py prepare        [--force]
    python3 scripts/workflow.py submit-relax   [--cases 1 3 5-8] [--dry-run] [--force]
    python3 scripts/workflow.py postprocess    [--cases ...] [--temps 300 600] [--nstep N] [--force]
    python3 scripts/workflow.py submit-nvt     [--cases ...] [--temps ...] [--dry-run] [--force]
    python3 scripts/workflow.py status         [--cases ...]
    python3 scripts/workflow.py restart relax|nvt [--cases ...] [--from-geometry] [--nstep N] [--dry-run]

--cases acepta numeros (3), rangos (5-8), listas (1,3,9) o trozos del nombre (alpha). Sin --cases: todos.
"""
import argparse
import configparser
import datetime
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import relax2aimd as r2a  # noqa: E402  (se reutilizan sus funciones de lectura/escritura de inputs)

RELAX_IN, RELAX_JOB, RELAX_OUT, RELAX_ERR = "input_relax.in", "job_relax.sh", "qe_out.out", "qe_out.err"
NVT_IN, NVT_JOB = "input_nvt_relax.in", "job_nvt.sh"      # mismo nombre que usabas en tu job.sh
PLACEHOLDERS = ("@JOB_NAME@", "@TIME@", "@OUT@", "@ERR@", "@INPUT@")
C = {}                                                    # configuracion (se rellena en main)


def die(msg):
    sys.exit("ERROR: " + msg)


# ------------------------------------------------------------------ configuracion
def load_cfg():
    path = ROOT / "workflow.cfg"
    if not path.is_file():
        die("no encuentro %s" % path)
    cp = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cp.read(str(path))

    def g(sec, key, default=""):
        return cp.get(sec, key, fallback=default).strip()

    C.update(
        structures=ROOT / g("paths", "structures", "structures"),
        cases=ROOT / g("paths", "cases", "cases"),
        pseudos=ROOT / g("paths", "pseudos", "pseudos"),
        dataset=ROOT / g("paths", "dataset", "dataset"),
        relax_in=ROOT / g("relax", "input_template", "templates/input_relax.in"),
        relax_job=ROOT / g("relax", "job_template", "templates/job_relax.sh"),
        relax_time=g("relax", "time", "24:00:00"),
        nvt_in=ROOT / g("nvt", "input_template", "templates/input_nvt.in"),
        nvt_job=ROOT / g("nvt", "job_template", "templates/job_nvt.sh"),
        nvt_time=g("nvt", "time", "72:00:00"),
        temps=[int(float(t)) for t in g("nvt", "temperatures", "1000").split()],
        nvt_nstep=g("nvt", "nstep", ""),
        sbatch=g("slurm", "sbatch", "sbatch"),
        squeue=g("slurm", "squeue", "squeue"),
    )


# ------------------------------------------------------------------ casos
def case_num(name):
    m = re.match(r"case_(\d+)_", name)
    return int(m.group(1)) if m else None


def case_id(name):
    n = case_num(name)
    return "c%02d" % n if n is not None else name


def select_cases(tokens):
    dirs = sorted(p for p in C["cases"].glob("case_*") if p.is_dir())
    if not dirs:
        die("no hay casos en %s (ejecuta antes 01_prepare.sh)" % C["cases"])
    if not tokens:
        return dirs
    sel = []
    for tok in tokens:
        for part in tok.split(","):
            part = part.strip()
            if not part:
                continue
            m = re.match(r"^(\d+)-(\d+)$", part)
            if m:
                nums = set(range(int(m.group(1)), int(m.group(2)) + 1))
            elif part.isdigit():
                nums = {int(part)}
            else:
                nums = None
            hit = [d for d in dirs if (nums is not None and case_num(d.name) in nums)
                   or (nums is None and part in d.name)]
            if not hit:
                print("   aviso: '%s' no coincide con ningun caso" % part)
            for d in hit:
                if d not in sel:
                    sel.append(d)
    if not sel:
        die("--cases no selecciona ningun caso")
    return sorted(sel)


# ------------------------------------------------------------------ lectura/escritura de inputs
def read_xyz(path):
    """extxyz sencillo -> (especies, posiciones[A], celda 3x3[A]). Acepta CRLF."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    m = re.search(r'Lattice\s*=\s*"([^"]+)"', lines[1], re.I)
    if not m:
        die("%s no trae Lattice=\"...\" en la cabecera." % path)
    v = [float(x) for x in m.group(1).split()]
    cell = [v[0:3], v[3:6], v[6:9]]
    sp, pos = [], []
    for ln in lines[2:2 + n]:
        t = ln.split()
        sp.append(t[0])
        pos.append([float(t[1]), float(t[2]), float(t[3])])
    if len(sp) != n:
        die("%s: esperaba %d atomos y he leido %d." % (path, n, len(sp)))
    return sp, pos, cell


def get_var(text, name):
    m = re.search(r"(?im)^[ \t]*%s[ \t]*=[ \t]*([^,\n!]*?)[ \t]*(?:,|!|$)" % re.escape(name), text)
    return m.group(1).strip().strip("'\"") if m else None


def set_var(text, name, value, insert_in_control=False):
    """Cambia 'name = valor' de una namelist (primera aparicion, linea propia). Conserva coma/comentario."""
    pat = re.compile(r"(?im)^(?P<pre>[ \t]*%s[ \t]*=[ \t]*)(?P<val>[^,\n!]*?)(?P<post>[ \t]*(?:,[ \t]*)?(?:!.*)?)$"
                     % re.escape(name))
    new, n = pat.subn(lambda m: m.group("pre") + value + m.group("post"), text, count=1)
    if n:
        return new
    if insert_in_control:
        return re.sub(r"(?im)^([ \t]*&control[ \t]*)$", lambda m: m.group(1) + "\n %s=%s," % (name, value), text, count=1)
    die("no encuentro '%s' en el input; anadelo al template." % name)


def validate_qe_input(path):
    """Comprobaciones basicas de coherencia (nat, especies, celda)."""
    lines = Path(path).read_text().splitlines()
    b = r2a.card_bounds(lines, "ATOMIC_POSITIONS")
    atoms = [l.split()[0] for l in lines[b[0] + 1:b[1]] if l.split()] if b else []
    nat = re.search(r"(?i)(?<![A-Za-z0-9_])nat\s*=\s*(\d+)", "\n".join(lines))
    labels = set(r2a.species_labels(lines))
    problems = []
    if not nat or int(nat.group(1)) != len(atoms):
        problems.append("nat=%s pero hay %d atomos en ATOMIC_POSITIONS" % (nat.group(1) if nat else "?", len(atoms)))
    if set(atoms) - labels:
        problems.append("especies sin ATOMIC_SPECIES: %s" % ", ".join(sorted(set(atoms) - labels)))
    if r2a.card_bounds(lines, "CELL_PARAMETERS") is None:
        problems.append("falta CELL_PARAMETERS")
    if problems:
        die("%s: %s" % (path, "; ".join(problems)))
    return len(atoms)


def pseudo_files(template_lines):
    names = []
    b = r2a.card_bounds(template_lines, "ATOMIC_SPECIES")
    if b is None:
        die("el template no tiene ATOMIC_SPECIES")
    for ln in template_lines[b[0] + 1:b[1]]:
        t = ln.split()
        if len(t) >= 3:
            names.append(t[2])
    return names


def copy_pseudos(names, dest):
    for n in names:
        src = C["pseudos"] / n
        if not src.is_file():
            die("falta el pseudopotencial %s (esperado en %s). Los nombres deben coincidir con ATOMIC_SPECIES."
                % (n, C["pseudos"]))
        shutil.copy2(str(src), str(dest / n))


def render_job(template, dest, **kw):
    text = Path(template).read_text()
    missing = [p for p in PLACEHOLDERS if p not in text]
    if missing:
        die("%s no contiene los marcadores %s" % (template, ", ".join(missing)))
    for k, v in kw.items():
        text = text.replace("@%s@" % k, str(v))
    Path(dest).write_text(text)


# ------------------------------------------------------------------ Slurm
def read_jobid(d):
    f = Path(d) / ".jobid"
    if f.is_file():
        t = f.read_text().split()
        return t[0] if t else None
    return None


def slurm_state(jobid):
    if not jobid:
        return None
    try:
        out = subprocess.check_output([C["squeue"], "-h", "-j", str(jobid), "-o", "%T"],
                                      stderr=subprocess.DEVNULL, universal_newlines=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    st = out.split()[0] if out else None
    # estados terminales (COMPLETED, TIMEOUT, FAILED...) = ya no activo: decide el contenido del output
    return st if st in ("PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED") else None


def sbatch(d, script, label, dry):
    if dry:
        print("   [dry-run] (cd %s && sbatch %s)" % (d, script))
        return None
    try:
        out = subprocess.check_output([C["sbatch"], script], cwd=str(d), universal_newlines=True,
                                      stderr=subprocess.STDOUT)
    except OSError:
        die("no encuentro '%s'. Ejecuta esto en el login node del cluster." % C["sbatch"])
    except subprocess.CalledProcessError as e:
        print("   FALLO sbatch en %s: %s" % (d, (e.output or "").strip()))
        return None
    m = re.search(r"Submitted batch job (\d+)", out)
    jid = m.group(1) if m else None
    if jid:
        (Path(d) / ".jobid").write_text(jid + "\n")
        with open(str(ROOT / "jobs.log"), "a") as fh:
            fh.write("%s  %-34s  job %s\n" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, jid))
    print("   enviado: job %s" % jid)
    return jid


# ------------------------------------------------------------------ lectura de outputs de QE
def scan_output(path):
    """Resumen de un output de pw.x (relax o md) sin cargarlo entero en memoria."""
    r = dict(exists=False, done=False, converged=False, maxsteps=False, error=None, scf=0, md_step=0, mtime=0)
    p = Path(path)
    if not p.is_file():
        return r
    r["exists"], r["mtime"] = True, p.stat().st_mtime
    with open(str(p), errors="replace") as fh:
        for ln in fh:
            if ln.startswith("!"):
                r["scf"] += 1
            elif "Entering Dynamics" in ln:
                m = re.search(r"iteration\s*=\s*(\d+)", ln)
                if m:
                    r["md_step"] = max(r["md_step"], int(m.group(1)))
            elif "JOB DONE" in ln:
                r["done"] = True
            elif "bfgs converged" in ln:
                r["converged"] = True
            elif "maximum number of steps has been reached" in ln:
                r["maxsteps"] = True
            elif "Error in routine" in ln and not r["error"]:
                r["error"] = ln.strip()
    return r


def relax_state(cd):
    rd = Path(cd) / "relax"
    if not (rd / RELAX_IN).is_file():
        return "SIN PREPARAR", ""
    sc = scan_output(rd / RELAX_OUT)
    st = slurm_state(read_jobid(rd))
    det = "scf=%d" % sc["scf"] if sc["exists"] else ""
    if st in ("PENDING", "CONFIGURING"):
        return "EN COLA", ""
    if st in ("RUNNING", "COMPLETING", "SUSPENDED"):
        return "EJECUTANDO", det
    if not sc["exists"]:
        return ("SIN SALIDA (cancelado?)", "") if read_jobid(rd) else ("PENDIENTE", "")
    if sc["error"]:
        return "ERROR QE", sc["error"]
    if sc["done"] and sc["converged"]:
        return "OK", det
    if sc["done"] and sc["maxsteps"]:
        return "NO CONVERGE (nstep)", det
    if sc["done"]:
        return "TERMINADO SIN CONVERGER", det
    if time.time() - sc["mtime"] < 900:
        return "ACTIVO? (sin squeue)", det
    return "INTERRUMPIDO", det


def nvt_outputs(nd):
    outs = []
    for f in Path(nd).glob("output.*"):
        m = re.match(r"output\.(\d+)$", f.name)
        if m:
            outs.append((int(m.group(1)), f))
    return sorted(outs)


def nvt_state(nd):
    nd = Path(nd)
    if not (nd / NVT_IN).is_file():
        return "SIN PREPARAR", ""
    nstep = int(float(get_var((nd / NVT_IN).read_text(), "nstep") or 0))
    outs = nvt_outputs(nd)
    steps, last = 0, None
    for _, f in outs:
        s = scan_output(f)
        steps, last = max(steps, s["md_step"]), s
    st = slurm_state(read_jobid(nd))
    det = "%d/%d pasos" % (steps, nstep)
    if st in ("PENDING", "CONFIGURING"):
        return "EN COLA", ""
    if st in ("RUNNING", "COMPLETING", "SUSPENDED"):
        return "EJECUTANDO", det
    if not outs:
        return ("SIN SALIDA (cancelado?)", "") if read_jobid(nd) else ("PENDIENTE", "")
    if last["error"]:
        return "ERROR QE", last["error"]
    if last["done"] and steps >= nstep:
        return "OK", det
    if last["done"]:
        return "TERMINADO (pasos<nstep)", det
    if time.time() - last["mtime"] < 900:
        return "ACTIVO? (sin squeue)", det
    return "INTERRUMPIDO", det


def nvt_dirs(cd, temps=None):
    dirs = sorted(Path(cd).glob("nvt_T*K"))
    if temps:
        dirs = [d for d in dirs if int(re.match(r"nvt_T(\d+)K", d.name).group(1)) in temps]
    return dirs


# ------------------------------------------------------------------ prepare
def cmd_prepare(args):
    tpl_lines = C["relax_in"].read_text().splitlines()
    labels = r2a.species_labels(tpl_lines)
    pnames = pseudo_files(tpl_lines)
    for n in pnames:
        if not (C["pseudos"] / n).is_file():
            die("falta el pseudopotencial %s en %s" % (n, C["pseudos"]))
    if r2a.card_bounds(tpl_lines, "CELL_PARAMETERS") is None or r2a.card_bounds(tpl_lines, "ATOMIC_POSITIONS") is None:
        die("el template de relax necesita CELL_PARAMETERS (ibrav=0) y ATOMIC_POSITIONS.")
    xyzs = sorted(C["structures"].glob("case_*.xyz"))
    if not xyzs:
        die("no hay case_*.xyz en %s" % C["structures"])
    print("%-34s %5s  %s" % ("caso", "nat", "celda (A)"))
    for xyz in xyzs:
        name = xyz.stem
        cd = C["cases"] / name
        rd = cd / "relax"
        if (rd / RELAX_OUT).is_file() and not args.force:
            print("%-34s  salto: ya tiene resultados (usa --force para regenerar el input)" % name)
            continue
        sp, pos, cell = read_xyz(xyz)
        bad = sorted(set(sp) - set(labels))
        if bad:
            die("%s: especies %s no estan en ATOMIC_SPECIES del template." % (xyz.name, ", ".join(bad)))
        lines = r2a.replace_card(tpl_lines, "ATOMIC_POSITIONS", r2a.fmt_positions(sp, pos))
        lines = r2a.replace_card(lines, "CELL_PARAMETERS", r2a.fmt_cell(cell))
        lines, _ = r2a.set_nat(lines, len(sp))
        text = "\n".join(lines) + "\n"
        text = set_var(text, "prefix", "'%s_relax'" % case_id(name))
        text = set_var(text, "restart_mode", "'from_scratch'", insert_in_control=True)
        rd.mkdir(parents=True, exist_ok=True)
        (rd / RELAX_IN).write_text(text)
        render_job(C["relax_job"], rd / RELAX_JOB, JOB_NAME="rlx_%s" % case_id(name), TIME=C["relax_time"],
                   OUT=RELAX_OUT, ERR=RELAX_ERR, INPUT=RELAX_IN)
        copy_pseudos(pnames, rd)
        shutil.copy2(str(xyz), str(cd / "initial.xyz"))
        validate_qe_input(rd / RELAX_IN)
        print("%-34s %5d  %.4f x %.4f x %.4f" % (name, len(sp), cell[0][0], (cell[1][0] ** 2 + cell[1][1] ** 2) ** .5,
                                                 cell[2][2]))
    print("\nListo. Revisa un caso (p. ej. cases/%s/relax/) y lanza ./02_submit_relax.sh" % xyzs[0].stem)


# ------------------------------------------------------------------ submit
def cmd_submit_relax(args):
    for cd in select_cases(args.cases):
        label, det = relax_state(cd)
        print("%-34s %s" % (cd.name, label))
        if label != "PENDIENTE" and not (args.force and label not in ("EN COLA", "EJECUTANDO", "SIN PREPARAR")):
            print("   salto (para continuar un relax interrumpido usa ./06_restart.sh relax)")
            continue
        sbatch(cd / "relax", RELAX_JOB, "%s relax" % cd.name, args.dry_run)


def cmd_submit_nvt(args):
    temps = args.temps
    for cd in select_cases(args.cases):
        dirs = nvt_dirs(cd, temps)
        if not dirs:
            print("%-34s sin carpetas NVT (ejecuta antes 03_postprocess_relax.sh)" % cd.name)
            continue
        for nd in dirs:
            label, det = nvt_state(nd)
            print("%-34s %-10s %s" % (cd.name, nd.name, label))
            if label != "PENDIENTE" and not (args.force and label not in ("EN COLA", "EJECUTANDO", "SIN PREPARAR")):
                print("   salto (para continuar una NVT interrumpida usa ./06_restart.sh nvt)")
                continue
            sbatch(nd, NVT_JOB, "%s %s" % (cd.name, nd.name), args.dry_run)


# ------------------------------------------------------------------ postprocess
def run_r2a(qe_out, qe_in, out_in, xyz_final, xyz_traj, cell_mode):
    cmd = [sys.executable, str(ROOT / "scripts" / "relax2aimd.py"), str(qe_out), str(qe_in), "-o", str(out_in),
           "--xyz-final", str(xyz_final), "--xyz-traj", str(xyz_traj), "--cell", cell_mode]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    out = p.communicate()[0]
    return p.returncode, out


def merge_xyz(parts, dest):
    with open(str(dest), "w") as fo:
        for p in parts:
            fo.write(Path(p).read_text())


def cmd_postprocess(args):
    temps = args.temps or C["temps"]
    nstep = args.nstep or C["nvt_nstep"]
    tpl_nvt = C["nvt_in"]
    nvt_lines = tpl_nvt.read_text().splitlines()
    pnames = pseudo_files(nvt_lines)
    done_n = 0
    for cd in select_cases(args.cases):
        rd = cd / "relax"
        label, _ = relax_state(cd)
        print("%-34s relax: %s" % (cd.name, label))
        usable = label == "OK" or (args.force and (rd / RELAX_OUT).is_file()
                                    and label not in ("EN COLA", "EJECUTANDO", "SIN PREPARAR"))
        if not usable:
            print("   salto (el relax no ha terminado bien; --force usa la ultima geometria disponible)")
            continue
        if any(nvt_outputs(nd) for nd in nvt_dirs(cd)) and not args.force:
            print("   salto: ya hay NVT con resultados (--force para regenerar los inputs)")
            continue
        base = rd / "relaxed_base.in"
        # El template NVT lleva la celda de otro sistema: se le pone la celda exacta de ESTE caso (la del input
        # de relax) para no heredar el redondeo de alat del output. Con --cell auto, relax2aimd solo la
        # cambiaria si el output trajera una celda distinta (p. ej. un vc-relax).
        rl = (rd / RELAX_IN).read_text().splitlines()
        cb = r2a.card_bounds(rl, "CELL_PARAMETERS")
        tmp_tpl = rd / "_nvt_template.in"
        tmp_tpl.write_text("\n".join(r2a.replace_card(nvt_lines, "CELL_PARAMETERS", rl[cb[0]:cb[1]])) + "\n")
        rc, out = run_r2a(rd / RELAX_OUT, tmp_tpl, base, rd / "relax_final.xyz", rd / "relax_traj.xyz", "auto")
        tmp_tpl.unlink()
        (rd / "postprocess.log").write_text(out)
        if rc != 0:
            print("   FALLO relax2aimd.py:\n      " + out.strip().replace("\n", "\n      "))
            continue
        for line in out.splitlines():
            if line.startswith(("Atomos", "Relax", "ATENCION", "Energia", "Trayectoria", "Celda")):
                print("   " + line)
        # trayectoria de los tramos previos (si el relax se reinicio) -> trayectoria completa
        parts = sorted(rd.glob("qe_out.part*.out"))
        traj_parts = []
        for k, pf in enumerate(parts, 1):
            tp = rd / ("relax_traj_part%d.xyz" % k)
            if not tp.is_file():
                rc2, o2 = run_r2a(pf, rd / RELAX_IN, rd / "_tmp.in", rd / ("relax_final_part%d.xyz" % k), tp, "input")
                if (rd / "_tmp.in").exists():
                    (rd / "_tmp.in").unlink()
                if rc2 != 0:
                    print("   aviso: no he podido leer %s (%s)" % (pf.name, o2.strip().splitlines()[-1] if o2.strip() else ""))
                    continue
            traj_parts.append(tp)
        traj = rd / "relax_traj.xyz"
        if traj_parts:
            traj = rd / "relax_traj_all.xyz"
            merge_xyz(traj_parts + [rd / "relax_traj.xyz"], traj)
            print("   trayectoria completa (%d tramos) en %s" % (len(traj_parts) + 1, traj.name))
        # copia al dataset
        ddir = C["dataset"] / "relax"
        ddir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(traj), str(ddir / (cd.name + "_traj.xyz")))
        shutil.copy2(str(rd / "relax_final.xyz"), str(ddir / (cd.name + "_final.xyz")))
        # carpetas NVT, una por temperatura
        base_text = base.read_text()
        for T in temps:
            nd = cd / ("nvt_T%dK" % T)
            nd.mkdir(exist_ok=True)
            text = set_var(base_text, "prefix", "'%s_T%dK'" % (case_id(cd.name), T))
            text = set_var(text, "tempw", "%.1f" % T)
            text = set_var(text, "restart_mode", "'from_scratch'", insert_in_control=True)
            if nstep:
                text = set_var(text, "nstep", str(int(nstep)))
            (nd / NVT_IN).write_text(text)
            render_job(C["nvt_job"], nd / NVT_JOB, JOB_NAME="nvt_%s_T%d" % (case_id(cd.name), T),
                       TIME=C["nvt_time"], OUT="output.1", ERR="error.1", INPUT=NVT_IN)
            copy_pseudos(pnames, nd)
            n = validate_qe_input(nd / NVT_IN)
            print("   NVT T=%d K: %s (nat=%d, nstep=%s)" % (T, (nd / NVT_IN).relative_to(ROOT), n,
                                                         get_var(text, "nstep")))
        done_n += 1
    print("\n%d caso(s) postprocesados. Lanza las dinamicas con ./04_submit_nvt.sh" % done_n)


# ------------------------------------------------------------------ status
def cmd_status(args):
    cases = select_cases(args.cases)
    print("%-34s %5s  %-34s %s" % ("caso", "nat", "relax", "nvt"))
    for cd in cases:
        nat = ""
        xi = cd / "initial.xyz"
        if xi.is_file():
            nat = xi.read_text().split("\n", 1)[0].strip()
        label, det = relax_state(cd)
        rtxt = label if label == "ERROR QE" or not det else "%s (%s)" % (label, det)
        dirs = nvt_dirs(cd)
        ntxt = "-"
        if dirs:
            items = []
            for nd in dirs:
                l2, d2 = nvt_state(nd)
                T = re.match(r"nvt_T(\d+)K", nd.name).group(1)
                items.append("T%s: %s%s" % (T, l2, " (%s)" % d2 if d2 and l2 != "ERROR QE" else ""))
            ntxt = " | ".join(items)
        print("%-34s %5s  %-34s %s" % (cd.name, nat, rtxt, ntxt))
        if label == "ERROR QE":
            print("%-34s %5s  -> %s" % ("", "", det))


# ------------------------------------------------------------------ restart
def next_run_index(nd):
    outs = nvt_outputs(nd)
    return (outs[-1][0] + 1) if outs else 1


def cmd_restart(args):
    ok_states = ("INTERRUMPIDO", "NO CONVERGE (nstep)", "TERMINADO (pasos<nstep)")
    for cd in select_cases(args.cases):
        if args.stage == "relax":
            rd = cd / "relax"
            label, _ = relax_state(cd)
            print("%-34s relax: %s" % (cd.name, label))
            if label not in ok_states and not (args.force and (rd / RELAX_OUT).is_file()
                                               and label not in ("EN COLA", "EJECUTANDO")):
                print("   salto (solo se reinicia si esta INTERRUMPIDO o NO CONVERGE; --force para forzar)")
                continue
            inp = rd / RELAX_IN
            text = inp.read_text()
            k = len(list(rd.glob("qe_out.part*.out"))) + 1
            prefix = get_var(text, "prefix")
            outdir = get_var(text, "outdir") or "./save"
            has_data = (rd / outdir / ("%s.save" % prefix)).is_dir()
            from_geo = args.from_geometry or not has_data
            if from_geo:
                if not args.from_geometry:
                    print("   no hay datos de restart en %s/%s.save -> se reinicia desde la ultima geometria" % (outdir, prefix))
                rc, out = run_r2a(rd / RELAX_OUT, inp, rd / "_new.in", rd / ("relax_final_part%d.xyz" % k),
                                  rd / ("relax_traj_part%d.xyz" % k), "input")
                if rc != 0:
                    print("   FALLO leyendo el output:\n      " + out.strip().replace("\n", "\n      "))
                    continue
                shutil.copy2(str(inp), str(rd / ("input_relax.in.part%d" % k)))
                new = (rd / "_new.in").read_text()
                (rd / "_new.in").unlink()
                new = set_var(new, "restart_mode", "'from_scratch'", insert_in_control=True)
            else:
                shutil.copy2(str(inp), str(rd / ("input_relax.in.part%d" % k)))
                new = set_var(text, "restart_mode", "'restart'", insert_in_control=True)
            if args.nstep:
                new = set_var(new, "nstep", str(int(args.nstep)))
            if args.dry_run:
                print("   [dry-run] modo: %s; se guardaria qe_out.part%d.out" % ("geometria" if from_geo else "restart QE", k))
                continue
            inp.write_text(new)
            for f in (RELAX_OUT, RELAX_ERR):
                if (rd / f).is_file():
                    shutil.move(str(rd / f), str(rd / f.replace(".out", ".part%d.out" % k).replace(".err", ".part%d.err" % k)))
            print("   modo: %s; tramo anterior guardado como qe_out.part%d.out" % ("geometria" if from_geo else "restart QE", k))
            sbatch(rd, RELAX_JOB, "%s relax (restart %d)" % (cd.name, k), args.dry_run)
        else:
            for nd in nvt_dirs(cd, args.temps):
                label, det = nvt_state(nd)
                print("%-34s %-10s %s %s" % (cd.name, nd.name, label, det))
                if label not in ok_states and not (args.force and nvt_outputs(nd) and label not in ("EN COLA", "EJECUTANDO")):
                    print("   salto (solo se reinicia si esta INTERRUMPIDO; --force para forzar)")
                    continue
                inp = nd / NVT_IN
                text = inp.read_text()
                prefix = get_var(text, "prefix")
                outdir = get_var(text, "outdir") or "./save"
                if not (nd / outdir / ("%s.save" % prefix)).is_dir():
                    print("   no hay datos de restart (%s/%s.save): no se puede continuar; borra la carpeta y repite el postprocess"
                          % (outdir, prefix))
                    continue
                run = next_run_index(nd)
                new = set_var(text, "restart_mode", "'restart'", insert_in_control=True)
                if args.nstep:
                    new = set_var(new, "nstep", str(int(args.nstep)))
                if args.dry_run:
                    print("   [dry-run] restart_mode='restart'; salida output.%d" % run)
                    continue
                shutil.copy2(str(inp), str(nd / (NVT_IN + ".run%d" % (run - 1))))
                inp.write_text(new)
                render_job(C["nvt_job"], nd / NVT_JOB, JOB_NAME="nvt_%s_T%s" % (case_id(cd.name), re.match(r"nvt_T(\d+)K", nd.name).group(1)),
                           TIME=C["nvt_time"], OUT="output.%d" % run, ERR="error.%d" % run, INPUT=NVT_IN)
                print("   restart_mode='restart'; la salida ira a output.%d" % run)
                sbatch(nd, NVT_JOB, "%s %s (restart %d)" % (cd.name, nd.name, run), args.dry_run)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.required = True

    def common(p, temps=False, dry=False, force=False):
        p.add_argument("--cases", nargs="*", default=None, help="numeros, rangos (5-8) o trozos del nombre")
        if temps:
            p.add_argument("--temps", nargs="*", type=int, default=None, help="temperaturas (K)")
        if dry:
            p.add_argument("--dry-run", action="store_true", help="no envia nada, solo muestra")
        if force:
            p.add_argument("--force", action="store_true")

    p = sub.add_parser("prepare"); p.add_argument("--force", action="store_true")
    p = sub.add_parser("submit-relax"); common(p, dry=True, force=True)
    p = sub.add_parser("postprocess"); common(p, temps=True, force=True)
    p.add_argument("--nstep", type=int, default=None, help="sobrescribe nstep de la NVT (p. ej. 5 para una prueba)")
    p = sub.add_parser("submit-nvt"); common(p, temps=True, dry=True, force=True)
    p = sub.add_parser("status"); common(p)
    p = sub.add_parser("restart"); p.add_argument("stage", choices=["relax", "nvt"]); common(p, temps=True, dry=True, force=True)
    p.add_argument("--from-geometry", action="store_true",
                   help="(relax) empieza un relax nuevo desde la ultima geometria del output en vez del restart de QE")
    p.add_argument("--nstep", type=int, default=None)
    args = ap.parse_args()

    load_cfg()
    {"prepare": cmd_prepare, "submit-relax": cmd_submit_relax, "postprocess": cmd_postprocess,
     "submit-nvt": cmd_submit_nvt, "status": cmd_status, "restart": cmd_restart}[args.cmd](args)


if __name__ == "__main__":
    main()
