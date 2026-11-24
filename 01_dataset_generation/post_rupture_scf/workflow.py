#!/usr/bin/env python3
"""Seguimiento de los SCF de QE en MN5: enviar | estado | recoger.

Solo usa la biblioteca estándar de Python (≥ 3.6): funciona en el nodo de login de MN5 sin módulos.
Se llama desde 02_enviar_scf.sh, 03_estado.sh y 04_recoger.sh.

Estados de un caso (casos/<caso>/):
  convergido       output.out con "!", "convergence has been achieved", fuerzas y tensiones
                   completas y JOB DONE
  en_cola          enviado y esperando (marca .en_cola con un trabajo vivo)
  en_marcha        calculándose (marca .en_marcha con un trabajo vivo)
  sin_empezar      sin output
  incompleto       output cortado (límite de tiempo, trabajo cancelado o nodo caído)
  no_convergido    "convergence NOT achieved" en los electron_maxstep pasos del input (queda fuera)
  fallo_mpi        pw.x no arrancó con todos los procesos MPI
  error            pw.x paró con "Error in routine ..."
"""
import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
CASOS = RAIZ / "casos"
RY = 13.605691932782346       # eV    (CODATA 2006, las mismas que ase.io.espresso)
BOHR = 0.5291772083535413     # Å
PENDIENTES = ("sin_empezar", "incompleto", "fallo_mpi")
ORDEN = ("convergido", "en_marcha", "en_cola", "sin_empezar", "incompleto",
         "no_convergido", "fallo_mpi", "error")

RE_NPROC = re.compile(r"(?:running on|Number of MPI processes:)\s+(\d+)")
RE_NAT = re.compile(r"number of atoms/cell\s*=\s*(\d+)")
RE_ITER = re.compile(r"convergence has been achieved in\s+(\d+)\s+iterations")
RE_E = re.compile(r"^!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", re.M)
RE_MAGT = re.compile(r"total magnetization\s+=\s+(-?\d+\.\d+)")
RE_MAGA = re.compile(r"absolute magnetization\s+=\s+(-?\d+\.\d+)")
RE_FUERZA = re.compile(r"atom\s+(\d+)\s+type\s+\d+\s+force\s+=\s+(\S+)\s+(\S+)\s+(\S+)")
RE_ERROR = re.compile(r"Error in routine\s+(.+?)\n(.*?)\n", re.S)
RE_WALL = re.compile(r"PWSCF\s+:.*?CPU\s+(.*?)\s+WALL")
RE_ITERACION = re.compile(r"^\s+iteration #\s*\d+", re.M)


# --------------------------------------------------------------------------- lectura
def segundos_wall(texto):
    if not texto:
        return None
    s = 0.0
    for valor, unidad in re.findall(r"([\d.]+)\s*([dhms])", texto):
        s += float(valor) * {"d": 86400, "h": 3600, "m": 60, "s": 1}[unidad]
    return s


def leer_output(path):
    """Lo que interesa de un output de pw.x (scf)."""
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    d = dict(fichero=Path(path).name)
    d["n_pwscf"] = txt.count("Program PWSCF")
    m = RE_NPROC.search(txt)
    d["nproc"] = int(m.group(1)) if m else None
    m = RE_NAT.search(txt)
    d["nat"] = int(m.group(1)) if m else None
    d["job_done"] = "JOB DONE" in txt
    d["no_conv"] = "convergence NOT achieved" in txt
    m = RE_ITER.search(txt)
    d["n_iter"] = int(m.group(1)) if m else None
    d["iteraciones"] = len(RE_ITERACION.findall(txt))
    e = RE_E.findall(txt)
    d["E_Ry"] = float(e[-1]) if e else None
    mt, ma = RE_MAGT.findall(txt), RE_MAGA.findall(txt)
    d["mag_tot"] = float(mt[-1]) if mt else None
    d["mag_abs"] = float(ma[-1]) if ma else None
    m = RE_ERROR.search(txt)
    d["error"] = (" ".join((m.group(1).strip().rstrip(":") + ": " + m.group(2).strip()).split())
                  if m else None)
    m = RE_WALL.search(txt)
    d["wall_s"] = segundos_wall(m.group(1)) if m else None
    # fuerzas (primer bloque tras "Forces acting on atoms"), Ry/bohr
    d["fuerzas"] = None
    i = txt.find("Forces acting on atoms")
    if i >= 0 and d["nat"]:
        f = []
        for linea in txt[i:].splitlines()[1:]:
            m = RE_FUERZA.search(linea)
            if m:
                f.append([float(m.group(k)) for k in (2, 3, 4)])
                if len(f) == d["nat"]:
                    break
            elif f:
                break
        d["fuerzas"] = f if len(f) == d["nat"] else None
    # tensión total, Ry/bohr³ (las 3 primeras columnas)
    d["tension"] = None
    i = txt.find("total   stress")
    if i >= 0:
        lineas = txt[i:].splitlines()[1:4]
        try:
            t = [[float(x) for x in ln.split()[:3]] for ln in lineas]
            if len(t) == 3 and all(len(r) == 3 for r in t):
                d["tension"] = t
        except ValueError:
            pass
    return d


def esperado_ntasks(carpeta):
    h = carpeta / "historial.txt"
    if not h.exists():
        return None
    n = re.findall(r"ntasks=(\d+)", h.read_text(encoding="utf-8"))
    return int(n[-1]) if n else None


def es_fallo_mpi(r, ntasks):
    return r["n_pwscf"] > 1 or (ntasks is not None and r["nproc"] is not None and r["nproc"] != ntasks)


def es_convergido(r, ntasks):
    return (r["E_Ry"] is not None and r["n_iter"] is not None and not r["no_conv"] and r["job_done"]
            and r["fuerzas"] is not None and r["tension"] is not None and not es_fallo_mpi(r, ntasks))


def trabajos_vivos():
    """{id: estado} de los trabajos del usuario en cola o en marcha (vacío si no hay squeue)."""
    try:
        out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i %T"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                             timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    vivos = {}
    for linea in out.stdout.split("\n"):
        p = linea.split()
        if len(p) >= 2:
            vivos[p[0].split("_")[0]] = p[1]
    return vivos


def estado_caso(carpeta, vivos):
    """(estado, resultado del output usado o None, detalle)."""
    for marca, etiqueta in ((".en_marcha", "en_marcha"), (".en_cola", "en_cola")):
        f = carpeta / marca
        if f.exists():
            jid = f.read_text(encoding="utf-8").strip()
            if vivos and jid in vivos:
                return etiqueta, None, f"trabajo {jid} ({vivos[jid]})"
    ntasks = esperado_ntasks(carpeta)
    if not (carpeta / "output.out").exists():
        return "sin_empezar", None, ""
    r = leer_output(carpeta / "output.out")
    if es_convergido(r, ntasks):
        return "convergido", r, f"{r['n_iter']} iter."
    if es_fallo_mpi(r, ntasks):
        return "fallo_mpi", r, f"{r['n_pwscf']} arranques de pw.x, {r['nproc']} procesos"
    if r["no_conv"]:
        return "no_convergido", r, f"{r['iteraciones']} iter. sin converger"
    if r["error"]:
        return "error", r, r["error"][:120]
    return "incompleto", r, f"output.out cortado tras {r['iteraciones']} iter."


def lista_casos():
    if not CASOS.is_dir():
        sys.exit(f"No existe {CASOS}: ejecuta antes 01_preparar_casos.ipynb y sube la carpeta.")
    return sorted(p.name for p in CASOS.iterdir() if (p / "input.in").exists())


def leer_indice():
    f = CASOS / "indice.csv"
    if not f.exists():
        return {}
    with open(f, newline="", encoding="utf-8") as fh:
        return {fila["caso"]: fila for fila in csv.DictReader(fh)}


def todos_los_estados(vivos):
    return {c: estado_caso(CASOS / c, vivos) for c in lista_casos()}


# --------------------------------------------------------------------------- enviar
def a_segundos(tiempo):
    dias = 0
    if "-" in tiempo:
        d, tiempo = tiempo.split("-", 1)
        dias = int(d)
    p = [int(x) for x in tiempo.split(":")]
    while len(p) < 3:
        p.insert(0, 0)
    return dias * 86400 + p[0] * 3600 + p[1] * 60 + p[2]


def cmd_enviar(a):
    vivos = trabajos_vivos()
    if vivos is None:
        print("aviso: squeue no responde; no puedo saber qué está en cola. Revisa antes de enviar.")
        vivos = {}
    estados = todos_los_estados(vivos)
    if a.casos:
        desconocidos = [c for c in a.casos if c not in estados]
        if desconocidos:
            sys.exit(f"Casos que no existen: {' '.join(desconocidos)}")
        lista = []
        for c in a.casos:
            e = estados[c][0]
            if e in ("en_cola", "en_marcha"):
                print(f"  {c}: {e}, se salta")
            elif e == "convergido" and not a.forzar:
                print(f"  {c}: ya convergido, se salta (usa --forzar para repetirlo)")
            else:
                lista.append(c)
    else:
        validos = set(PENDIENTES)
        if a.incluir_no_convergidos:
            validos.add("no_convergido")
        if a.incluir_errores:
            validos.add("error")
        if a.solo_control:
            lista = [c for c, (e, _, _) in estados.items() if e in validos and c.startswith("ctrl_")]
        else:
            # primero los de control (para comprobar la coherencia cuanto antes)
            lista = sorted((c for c, (e, _, _) in estados.items() if e in validos),
                           key=lambda c: (not c.startswith("ctrl_"), c))
    k, tiempo, qos = a.k, a.tiempo, a.qos
    if a.prueba:
        lista, k, tiempo, qos = lista[:1], 1, "02:00:00", "gp_debug"
    if not lista:
        print("No hay casos que enviar.")
        cmd_estado(argparse.Namespace(detalle=False, csv=False), vivos=vivos, estados=estados)
        return
    lotes = [lista[i:i + k] for i in range(0, len(lista), k)]
    if a.max_trabajos:
        lotes = lotes[:a.max_trabajos]
    (RAIZ / "logs").mkdir(exist_ok=True)
    limite = a_segundos(tiempo)
    if a.est_min * 60 + 900 > limite:
        sys.exit(f"--tiempo {tiempo} no deja hueco para un caso ({a.est_min} min estimados + 15 min de "
                 "margen): sube --tiempo o baja --est-min.")
    print(f"Envío: {sum(len(l) for l in lotes)} casos en {len(lotes)} trabajos de hasta {k} casos, "
          f"--time={tiempo}{f', --qos={qos}' if qos else ''}"
          f"{', FORZAR=1' if a.forzar else ''}")
    for lote in lotes:
        export = f"ALL,LIMITE_S={limite},FORZAR={1 if a.forzar else 0},EST_S={a.est_min * 60}"
        if a.pw_flags is not None:
            export += f",PW_FLAGS={a.pw_flags}"
        cmd = ["sbatch", f"--time={tiempo}", f"--job-name=scf_{lote[0]}", f"--export={export}"]
        if qos:
            cmd.append(f"--qos={qos}")
        cmd += ["job_scf.sh"] + lote
        if a.dry_run:
            print("  " + " ".join(cmd))
            continue
        out = subprocess.run(cmd, cwd=str(RAIZ), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True)
        m = re.search(r"Submitted batch job (\d+)", out.stdout)
        if not m:
            print(f"  ERROR al enviar {lote[0]}…: {out.stdout.strip()} {out.stderr.strip()}")
            print("  Se para aquí (¿límite de trabajos en cola? mira 'bsc_queues'). Los enviados siguen.")
            break
        jid = m.group(1)
        for c in lote:
            (CASOS / c / ".en_cola").write_text(jid + "\n", encoding="utf-8")
        print(f"  trabajo {jid}: {' '.join(lote)}")
    if a.dry_run:
        print("(--dry-run: no se ha enviado nada)")


# --------------------------------------------------------------------------- estado
def mediana(v):
    v = sorted(v)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def cmd_estado(a, vivos=None, estados=None):
    if vivos is None:
        vivos = trabajos_vivos()
        if vivos is None:
            print("aviso: squeue no responde; los casos en cola o en marcha saldrán como incompletos.")
            vivos = {}
    if estados is None:
        estados = todos_los_estados(vivos)
    conteo = {e: [] for e in ORDEN}
    for c, (e, _, _) in estados.items():
        conteo[e].append(c)
    n = len(estados)
    print(f"\n{n} casos en {CASOS}")
    for e in ORDEN:
        if conteo[e]:
            muestra = " ".join(conteo[e][:8]) + (" …" if len(conteo[e]) > 8 else "")
            print(f"  {e:<16s}{len(conteo[e]):5d}  ({100 * len(conteo[e]) / n:4.0f} %)   {muestra}")
    conv = [estados[c][1] for c in conteo["convergido"]]
    if conv:
        walls = [r["wall_s"] for r in conv if r["wall_s"]]
        iters = [r["n_iter"] for r in conv]
        mags = [abs(r["mag_tot"]) for r in conv if r["mag_tot"] is not None]
        texto_wall = (f"tiempo/caso: mediana {mediana(walls) / 60:.0f} min, máx {max(walls) / 60:.0f} min; "
                      if walls else "")
        print(f"\nConvergidos: {texto_wall}iteraciones: mediana {mediana(iters):.0f}, máx {max(iters)}; "
              f"|magnetización total| > 0.1 μB en "
              f"{sum(1 for m in mags if m > 0.1)}")
    trabajos = sorted({estados[c][2].split()[1] for c in conteo["en_marcha"] + conteo["en_cola"]})
    if trabajos:
        print(f"Trabajos vivos con casos: {len(trabajos)} ({' '.join(trabajos[:10])}{' …' if len(trabajos) > 10 else ''})")
    problemas = conteo["fallo_mpi"] + conteo["error"]
    for c in problemas:
        print(f"  {c}: {estados[c][0]} — {estados[c][2]}")
    if conteo["fallo_mpi"]:
        print("  → fallo_mpi: pw.x arrancó mal (srun lanzó copias en serie o con menos procesos). "
              "Revisa logs/ antes de reenviar.")
    pendientes = sum(len(conteo[e]) for e in PENDIENTES)
    if pendientes:
        print(f"\nPendientes de enviar: {pendientes} → ./02_enviar_scf.sh")
    if conteo["no_convergido"]:
        print(f"No convergidos: {len(conteo['no_convergido'])} (quedan fuera del dataset; para "
              "insistir, edita su input.in y reenvíalo con ./02_enviar_scf.sh --casos <caso>)")
    if getattr(a, "detalle", False):
        print(f"\n{'caso':<12s}{'estado':<17s}{'iter':>5s}{'min':>6s}{'E (eV/át)':>14s}{'mag':>7s}  detalle")
        for c, (e, r, det) in estados.items():
            if r:
                eat = (f"{r['E_Ry'] * RY / r['nat']:14.5f}" if r["E_Ry"] is not None and r["nat"] else f"{'':14s}")
                print(f"{c:<12s}{e:<17s}{(r['n_iter'] or r['iteraciones']):5d}"
                      f"{(r['wall_s'] or 0) / 60:6.0f}{eat}"
                      f"{(r['mag_tot'] if r['mag_tot'] is not None else float('nan')):7.2f}  {det}")
            else:
                print(f"{c:<12s}{e:<17s}{'':5s}{'':6s}{'':14s}{'':7s}  {det}")
    if getattr(a, "csv", False):
        with open(RAIZ / "estado.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["caso", "estado", "output", "n_iter", "iteraciones", "wall_s", "E_eV_at",
                        "mag_tot", "mag_abs", "detalle"])
            for c, (e, r, det) in estados.items():
                if r:
                    eat = r["E_Ry"] * RY / r["nat"] if r["E_Ry"] is not None and r["nat"] else ""
                    w.writerow([c, e, r["fichero"], r["n_iter"], r["iteraciones"], r["wall_s"], eat,
                                r["mag_tot"], r["mag_abs"], det])
                else:
                    w.writerow([c, e, "", "", "", "", "", "", "", det])
        print(f"\nEscrito {RAIZ / 'estado.csv'}")


# --------------------------------------------------------------------------- recoger
def leer_input(path):
    """Celda, símbolos y posiciones (Å) del input.in escrito por 01_preparar_casos."""
    lineas = Path(path).read_text(encoding="utf-8").splitlines()
    nat = int(re.search(r"\bnat\s*=\s*(\d+)", "\n".join(lineas)).group(1))
    celda, simbolos, pos = None, [], []
    for i, ln in enumerate(lineas):
        if ln.strip().upper().startswith("CELL_PARAMETERS"):
            if "angstrom" not in ln.lower():
                raise ValueError(f"{path}: CELL_PARAMETERS no está en angstrom")
            celda = [[float(x) for x in lineas[i + j].split()[:3]] for j in (1, 2, 3)]
        if ln.strip().upper().startswith("ATOMIC_POSITIONS"):
            if "angstrom" not in ln.lower():
                raise ValueError(f"{path}: ATOMIC_POSITIONS no está en angstrom")
            for j in range(1, nat + 1):
                p = lineas[i + j].split()
                simbolos.append(p[0])
                pos.append([float(x) for x in p[1:4]])
    return celda, simbolos, pos


def volumen(c):
    return abs(c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1])
               - c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0])
               + c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0]))


def valor_info(v):
    s = str(v)
    if s == "" or s.lower() == "nan":
        return None
    try:
        float(s)
        return s
    except ValueError:
        return s if re.fullmatch(r"[A-Za-z0-9_.+\-]+", s) else '"' + s.replace('"', "") + '"'


def json33(m):
    return '"_JSON [' + ", ".join("[" + ", ".join(f"{x:.10e}" for x in fila) + "]" for fila in m) + ']"'


def configuracion_extxyz(caso, r, fila_indice, config_type):
    celda, simb, pos = leer_input(CASOS / caso / "input.in")
    nat = len(simb)
    if nat != r["nat"]:
        raise ValueError(f"{caso}: {nat} átomos en input.in y {r['nat']} en el output")
    E = r["E_Ry"] * RY
    F = [[x * RY / BOHR for x in f] for f in r["fuerzas"]]
    sigma = [[-x * RY / BOHR ** 3 for x in fila] for fila in r["tension"]]   # signo ASE (tracción > 0)
    V = volumen(celda)
    vir = [[-x * V for x in fila] for fila in sigma]                          # virial = −V·σ
    info = {"energy_qe": f"{E:.10f}", "stress_qe": json33(sigma), "virials_qe": json33(vir),
            "config_type": config_type, "caso": caso, "n_iter": r["n_iter"],
            "mag_tot": r["mag_tot"], "mag_abs": r["mag_abs"]}
    for clave in ("traj", "frame", "t_ps", "T_obj_K", "T_inst_K", "BC_intactos", "area_A2_at", "z_rms_A",
                  "espesor_A", "ps_antes_colapso", "fps_rank"):
        if fila_indice and clave in fila_indice:
            info[clave] = fila_indice[clave]
    cab = 'Lattice="' + " ".join(f"{x:.10f}" for v in celda for x in v) + '" '
    cab += "Properties=species:S:1:pos:R:3:forces_qe:R:3 "
    for k, v in info.items():
        v = valor_info(v) if not str(v).startswith('"_JSON') else v
        if v is not None:
            cab += f"{k}={v} "
    cab += 'pbc="T T T"'
    cuerpo = [f"{s:<2s} {p[0]:16.10f} {p[1]:16.10f} {p[2]:16.10f} {f[0]:16.10f} {f[1]:16.10f} {f[2]:16.10f}"
              for s, p, f in zip(simb, pos, F)]
    fmax = max(math.sqrt(sum(x * x for x in f)) for f in F)
    return "\n".join([str(nat), cab] + cuerpo) + "\n", dict(E_eV_at=E / nat, Fmax=fmax, F=F, nat=nat)


def cmd_recoger(a):
    vivos = trabajos_vivos() or {}
    estados = todos_los_estados(vivos)
    indice = leer_indice()
    salida = RAIZ / a.salida
    salida.mkdir(exist_ok=True)
    bloques, resumen, controles, fuera = {}, [], [], []
    for c, (e, r, det) in estados.items():
        if e != "convergido":
            fuera.append((c, e, det))
            continue
        fila = indice.get(c, {})
        es_control = c.startswith("ctrl_") or fila.get("tipo") == "control"
        texto, x = configuracion_extxyz(c, r, fila, "control" if es_control else a.config_type)
        resumen.append(dict(caso=c, tipo="control" if es_control else "postrotura", output=r["fichero"],
                            n_iter=r["n_iter"], wall_s=r["wall_s"], E_eV_at=f"{x['E_eV_at']:.6f}",
                            Fmax_eV_A=f"{x['Fmax']:.3f}", mag_tot=r["mag_tot"], mag_abs=r["mag_abs"]))
        if es_control:
            ref_f = CASOS / c / "referencia.json"
            if ref_f.exists():
                ref = json.loads(ref_f.read_text(encoding="utf-8"))
                dE = (x["E_eV_at"] - ref["energy_eV"] / x["nat"]) * 1000
                dF = [fi - fr for f, g in zip(x["F"], ref["forces_eV_A"]) for fi, fr in zip(f, g)]
                rmse = 1000 * math.sqrt(sum(d * d for d in dF) / len(dF))
                controles.append((c, ref.get("config_type", ""), dE, rmse, max(abs(d) for d in dF) * 1000))
            bloques.setdefault("control", []).append(texto)
        else:
            bloques.setdefault("postrotura", []).append((c, texto))

    post = bloques.get("postrotura", [])
    n_test = int(round(a.test * len(post)))
    azar = random.Random(a.semilla)
    test_ids = set(azar.sample([c for c, _ in post], n_test)) if n_test else set()
    ficheros = {}
    if post:
        ficheros["todas"] = salida / f"{a.prefijo}_todas.extxyz"
        ficheros["todas"].write_text("".join(t for _, t in post), encoding="utf-8")
        if n_test:
            ficheros["train"] = salida / f"{a.prefijo}_train.extxyz"
            ficheros["test"] = salida / f"{a.prefijo}_test.extxyz"
            ficheros["train"].write_text("".join(t for c, t in post if c not in test_ids), encoding="utf-8")
            ficheros["test"].write_text("".join(t for c, t in post if c in test_ids), encoding="utf-8")
    if bloques.get("control"):
        ficheros["control"] = salida / f"{a.prefijo}_control.extxyz"
        ficheros["control"].write_text("".join(bloques["control"]), encoding="utf-8")
    with open(salida / f"{a.prefijo}_resumen.csv", "w", newline="", encoding="utf-8") as fh:
        campos = ["caso", "tipo", "output", "n_iter", "wall_s", "E_eV_at", "Fmax_eV_A", "mag_tot", "mag_abs", "split"]
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        for fila in resumen:
            fila["split"] = ("control" if fila["tipo"] == "control"
                             else ("test" if fila["caso"] in test_ids else "train"))
            w.writerow(fila)
    with open(salida / f"{a.prefijo}_fuera.txt", "w", encoding="utf-8") as fh:
        for c, e, det in fuera:
            fh.write(f"{c}\t{e}\t{det}\n")

    n_post = len(post)
    n_ctrl = len(bloques.get("control", []))
    total_post = sum(1 for c in estados if not c.startswith("ctrl_"))
    print(f"\nRecogidas {n_post} de {total_post} configuraciones (las convergidas)"
          + (f" + {n_ctrl} de control" if n_ctrl else ""))
    for k, f in ficheros.items():
        print(f"  {f.relative_to(RAIZ)}")
    print(f"  {(salida / (a.prefijo + '_resumen.csv')).relative_to(RAIZ)}, "
          f"{(salida / (a.prefijo + '_fuera.txt')).relative_to(RAIZ)} ({len(fuera)} fuera)")
    if n_test:
        print(f"  test: {n_test} configuraciones al azar (semilla {a.semilla}); train: {n_post - n_test}")
    if resumen:
        fmax = max(float(f["Fmax_eV_A"]) for f in resumen)
        mags = [abs(f["mag_tot"]) for f in resumen if f["mag_tot"] is not None]
        print(f"  |F| máx = {fmax:.1f} eV/Å; |magnetización total| > 0.1 μB en "
              f"{sum(1 for m in mags if m > 0.1)} de {len(mags)}")
    if controles:
        print("\nControl (mismos ajustes que la selección frente a la energía y fuerzas del train):")
        for c, ct, dE, rmse, dmax in controles:
            print(f"  {c} ({ct}): ΔE = {dE:+.3f} meV/át, RMSE F = {rmse:.1f} meV/Å, |ΔF| máx = {dmax:.1f} meV/Å")
        print("  Coherente si |ΔE| ≲ 0.5 meV/át y RMSE F ≲ 5 meV/Å; si no, los ajustes no son los del train.")
    print("\nClaves para MACE: energy_key=energy_qe, forces_key=forces_qe, stress_key=stress_qe "
          "(eV/Å³, signo ASE), virials_key=virials_qe (eV).")


# --------------------------------------------------------------------------- main
def salida_segura():
    """Si la terminal no es UTF-8 (locale C/ASCII), los caracteres raros salen como '?' en vez de dar error."""
    enc = (getattr(sys.stdout, "encoding", None) or "").lower().replace("-", "").replace("_", "")
    if enc == "utf8":
        return
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:                       # Python 3.6
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=sys.stdout.encoding or "ascii",
                                      errors="replace", line_buffering=True)


def main():
    salida_segura()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="orden")
    e = sub.add_parser("enviar", help="envía los casos pendientes")
    e.add_argument("--k", type=int, default=4, help="casos por trabajo (4)")
    e.add_argument("--tiempo", default="08:00:00", help="--time de cada trabajo (08:00:00)")
    e.add_argument("--qos", default=None, help="otra QoS (por defecto la de job_scf.sh: gp_ehpc)")
    e.add_argument("--max-trabajos", type=int, default=0, help="como mucho N trabajos (0 = todos)")
    e.add_argument("--casos", nargs="+", help="solo estos casos")
    e.add_argument("--prueba", action="store_true", help="un caso en gp_debug (2 h) para medir el tiempo")
    e.add_argument("--solo-control", action="store_true", help="solo los casos de control")
    e.add_argument("--forzar", action="store_true", help="repite desde cero aunque hayan convergido")
    e.add_argument("--incluir-no-convergidos", action="store_true", help="reenvía también los no convergidos")
    e.add_argument("--incluir-errores", action="store_true", help="reenvía los que pararon con error")
    e.add_argument("--pw-flags", default=None, help="opciones de pw.x (por defecto '-nk 2')")
    e.add_argument("--est-min", type=int, default=60, help="minutos estimados por caso hasta medir (60)")
    e.add_argument("--dry-run", action="store_true", help="muestra los sbatch sin enviarlos")
    s = sub.add_parser("estado", help="resumen de los casos")
    s.add_argument("--detalle", action="store_true", help="una línea por caso")
    s.add_argument("--csv", action="store_true", help="escribe estado.csv")
    r = sub.add_parser("recoger", help="junta los convergidos en extxyz para MACE")
    r.add_argument("--salida", default="dataset")
    r.add_argument("--prefijo", default="postrotura_scf")
    r.add_argument("--config-type", default="postrotura_FT")
    r.add_argument("--test", type=float, default=0.1, help="fracción para test (0.1; 0 = sin test)")
    r.add_argument("--semilla", type=int, default=42)
    a = p.parse_args()
    if a.orden == "enviar":
        cmd_enviar(a)
    elif a.orden == "estado":
        cmd_estado(a)
    elif a.orden == "recoger":
        cmd_recoger(a)
    else:
        p.print_help()


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:              # salida cortada con | head o | less: sin traceback
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
