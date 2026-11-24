"""
Continuacion de la MD NVT (Nose-Hoover chain) lanzada con md_nvt.py.

Lee el ultimo fotograma de la .traj y sigue la dinamica desde ahi, sin tocarla:
  - mismas posiciones y momentos (float64, tal como se guardaron);
  - mismo integrador y parametros (NoseHooverChainNVT, T, dt, tdamp, tchain, tloop);
  - mismo contador de pasos: los fotogramas nuevos se anaden a la MISMA .traj cada
    SAVE_INTERVAL pasos, asi que el fotograma i sigue siendo el paso i*SAVE_INTERVAL;
  - sin relajar, sin MaxwellBoltzmann, sin Stationary/ZeroRotation y sin borrar nada.

Termostato: md_nvt.py no guardaba las variables de la cadena (eta, p_eta), asi que el
PRIMER reinicio las pone a cero (igual que cuando empezo la MD). Este script si las
guarda en cada fotograma (atoms.info), y a partir de ahi cada reinicio es exacto.

Se para solo antes del limite de tiempo del trabajo, justo despues de guardar un
fotograma, para no perder pasos. Para seguir, vuelve a lanzar:
    sbatch submit_continuar.sh
Si ya se han hecho los NSTEPS, termina sin hacer nada (se puede encadenar sin riesgo).
"""

import os
import shutil
import sys
import time

import ase
import numpy as np
import torch
from ase import units
from ase.calculators.calculator import all_properties
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import Trajectory, write
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from mace.calculators import MACECalculator

# ---------------- Parametros (los mismos que en md_nvt.py) ----------------
MODEL_PATH    = "FT_MACE_v31_run-165.model"
SIZE          = "10x10"
TEMP          = 1500        # K
TIMESTEP_FS   = 1.0
TDAMP_FS      = 100.0       # valor por defecto de NoseHooverMD en md_nvt.py
TCHAIN        = 3           # valor por defecto de ASE (md_nvt.py no lo cambiaba)
TLOOP         = 1           # idem
NSTEPS        = 1000000     # pasos TOTALES de la MD (contando los ya hechos)
SAVE_INTERVAL = 1000        # pasos entre fotogramas de la .traj (no cambiar)
LOG_INTERVAL  = 1000        # pasos entre lineas del log termodinamico
EXPORTAR_XYZ  = True        # al llegar a NSTEPS, exporta la trayectoria entera a .xyz
MARGEN_MIN    = 10          # minutos de reserva al final de cada trabajo
WALLTIME_H    = 71.5        # solo si no se sabe la hora de fin del trabajo (ver abajo)
DEVICE        = "cuda"
# ---------------------------------------------------------------------------

FNAME       = f"md_bgdy_{SIZE}_{TEMP}_{TIMESTEP_FS:g}fs"
TRAJ_FILE   = FNAME + ".traj"
THERMO_FILE = FNAME + "_thermo.dat"     # se escribe linea a linea (no se pierde si se para)
THERMO_COLS = "step time_ps T_K Epot Ekin Etot Econs"   # energias en eV/atomo

T_INICIO = time.time()


def hora_fin_trabajo():
    """Hora (epoch) a la que Slurm matara el trabajo."""
    valor = os.environ.get("MD_FIN_EPOCH", "")
    try:
        fin = float(valor)
        if fin > T_INICIO:
            return fin, "Slurm"
    except ValueError:
        pass
    return T_INICIO + WALLTIME_H * 3600, f"WALLTIME_H = {WALLTIME_H} h"


def leer_ultimo_fotograma():
    if not os.path.isfile(TRAJ_FILE):
        sys.exit(f"No encuentro {TRAJ_FILE} en {os.getcwd()}")
    tr = Trajectory(TRAJ_FILE)
    n = len(tr)
    if n == 0:
        sys.exit(f"{TRAJ_FILE} no tiene fotogramas")
    atoms = tr[-1]
    tr.close()
    paso_esperado = (n - 1) * SAVE_INTERVAL
    paso = int(atoms.info.get("md_step", paso_esperado))
    if paso != paso_esperado:
        sys.exit(f"El ultimo fotograma dice paso {paso}, pero con {n} fotogramas "
                 f"cada {SAVE_INTERVAL} pasos deberia ser {paso_esperado}. Revisa la .traj.")
    if not atoms.has("momenta"):
        sys.exit("El ultimo fotograma no tiene momentos: no se puede continuar la MD.")
    return atoms, n, paso


def preparar_thermo(paso):
    """Deja THERMO_FILE con los pasos <= paso.

    Si no existe (primer reinicio), lo reconstruye a partir de los fotogramas de la
    .traj: con LOG_INTERVAL = SAVE_INTERVAL da exactamente lo que habria guardado
    log_thermo de md_nvt.py (T, Epot, Ekin, Etot); Econs no se guardaba -> nan.
    """
    if os.path.isfile(THERMO_FILE):
        with open(THERMO_FILE) as f:
            lineas = f.readlines()
        quedan = [l for l in lineas if l.startswith("#") or int(l.split()[0]) <= paso]
        if len(quedan) != len(lineas):
            with open(THERMO_FILE, "w") as f:
                f.writelines(quedan)
            print(f"  {THERMO_FILE}: quitadas {len(lineas) - len(quedan)} lineas de pasos > {paso}")
        return
    with Trajectory(TRAJ_FILE) as tr, open(THERMO_FILE, "w") as f:
        f.write(f"# {THERMO_COLS}\n")
        for i, a in enumerate(tr):
            step = i * SAVE_INTERVAL
            if step % LOG_INTERVAL:
                continue
            nat = len(a)
            epot = a.get_potential_energy() / nat
            ekin = a.get_kinetic_energy() / nat
            f.write(f"{step:d} {step * TIMESTEP_FS / 1000:.3f} {a.get_temperature():.4f} "
                    f"{epot:.10f} {ekin:.10f} {epot + ekin:.10f} nan\n")
    print(f"  {THERMO_FILE}: reconstruido desde la .traj (pasos 0-{paso})")


def exportar_xyz():
    """extxyz con celda, pbc, momentos, energia y fuerzas de cada fotograma."""
    def fotogramas():
        with Trajectory(TRAJ_FILE) as tr:
            for i, a in enumerate(tr):
                a.info = {"md_step": i * SAVE_INTERVAL,
                          "time_ps": i * SAVE_INTERVAL * TIMESTEP_FS / 1000}
                yield a
    write(FNAME + ".xyz", fotogramas(), format="extxyz")


def main():
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU not available!")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | PyTorch {torch.__version__} | "
              f"ASE {ase.__version__}", flush=True)

    conf, n_frames, paso = leer_ultimo_fotograma()
    print(f"{TRAJ_FILE}: {n_frames} fotogramas, ultimo = paso {paso} "
          f"({paso * TIMESTEP_FS / 1000:.1f} ps), T = {conf.get_temperature():.1f} K", flush=True)

    if paso >= NSTEPS:
        print(f"La MD ya llego a NSTEPS = {NSTEPS}. Nada que hacer.")
        if EXPORTAR_XYZ and not os.path.isfile(FNAME + ".xyz"):
            exportar_xyz()
            print(f"Exportado {FNAME}.xyz")
        return

    shutil.copy2(TRAJ_FILE, TRAJ_FILE + ".bak")
    print(f"  copia de seguridad: {TRAJ_FILE}.bak")
    preparar_thermo(paso)

    # ── Calculador e integrador (los mismos que en md_nvt.py) ─────────────
    conf.calc = MACECalculator(model_paths=MODEL_PATH, device=DEVICE, default_dtype="float64")
    dyn = NoseHooverChainNVT(
        atoms=conf,
        timestep=TIMESTEP_FS * units.fs,
        temperature_K=TEMP,
        tdamp=TDAMP_FS * units.fs,
        tchain=TCHAIN,
        tloop=TLOOP,
    )
    dyn.nsteps = paso       # el contador sigue donde se quedo

    termo = getattr(dyn, "_thermostat", None)
    if termo is None or not hasattr(termo, "_eta") or not hasattr(termo, "_p_eta"):
        raise RuntimeError(f"ASE {ase.__version__}: NoseHooverChainNVT no tiene _thermostat._eta/"
                           "_p_eta. Este script esta hecho para ASE 3.27 (la version de la .traj).")

    if "nhc_p_eta" in conf.info:
        for clave, valor in [("nhc_T_K", TEMP), ("nhc_tdamp_fs", TDAMP_FS), ("dt_fs", TIMESTEP_FS)]:
            if clave in conf.info and not np.isclose(conf.info[clave], valor):
                sys.exit(f"{clave} = {conf.info[clave]} en la .traj, pero {valor} en este script.")
        eta = np.asarray(conf.info["nhc_eta"], dtype=float)
        p_eta = np.asarray(conf.info["nhc_p_eta"], dtype=float)
        if eta.shape != (TCHAIN,) or p_eta.shape != (TCHAIN,):
            sys.exit(f"La .traj tiene una cadena de {eta.size} termostatos y TCHAIN = {TCHAIN}.")
        termo._eta[:] = eta
        termo._p_eta[:] = p_eta
        print("  reinicio EXACTO: posiciones, momentos y termostato (eta, p_eta) del ultimo fotograma")
    else:
        print("  primer reinicio desde md_nvt.py: posiciones y momentos del ultimo fotograma; "
              "la .traj no tiene el termostato -> eta = p_eta = 0 (como al empezar la MD)")

    # Los atomos de la MD sin info, como en md_nvt.py. MACE trata un cambio en atoms.info
    # como un cambio del sistema (borraria la energia y las fuerzas ya calculadas).
    conf.info = {}

    # ── Observadores ──────────────────────────────────────────────────────
    traj = Trajectory(TRAJ_FILE, "a", conf)

    def guardar_fotograma():
        # Se guarda una copia: mismas posiciones, momentos, energia, fuerzas y stress que
        # escribiria traj.write, mas en info lo necesario para un reinicio exacto.
        conf.get_forces()                      # ya calculadas en este paso (cache)
        a = conf.copy()
        a.info = {
            "md_step": dyn.nsteps,
            "time_ps": dyn.nsteps * TIMESTEP_FS / 1000,
            "nhc_eta": termo._eta.copy(),
            "nhc_p_eta": termo._p_eta.copy(),
            "nhc_T_K": float(TEMP),
            "nhc_tdamp_fs": float(TDAMP_FS),
            "dt_fs": float(TIMESTEP_FS),
        }
        a.calc = SinglePointCalculator(
            a, **{k: v for k, v in conf.calc.results.items() if k in all_properties})
        traj.write(a)

    fth = open(THERMO_FILE, "a")
    reloj = {"t": time.perf_counter(), "paso": dyn.nsteps}

    def log_thermo():
        nat = len(conf)
        epot = conf.get_potential_energy() / nat
        ekin = conf.get_kinetic_energy() / nat
        econs = dyn.get_conserved_energy() / nat
        temp = conf.get_temperature()
        fth.write(f"{dyn.nsteps:d} {dyn.nsteps * TIMESTEP_FS / 1000:.3f} {temp:.4f} "
                  f"{epot:.10f} {ekin:.10f} {epot + ekin:.10f} {econs:.10f}\n")
        fth.flush()
        ahora = time.perf_counter()
        s_paso = (ahora - reloj["t"]) / max(dyn.nsteps - reloj["paso"], 1)
        reloj["t"], reloj["paso"] = ahora, dyn.nsteps
        print(f"paso {dyn.nsteps:>7d}/{NSTEPS}  T = {temp:7.1f} K  ({s_paso:.3f} s/paso)", flush=True)

    dyn.attach(guardar_fotograma, interval=SAVE_INTERVAL)
    dyn.attach(log_thermo, interval=LOG_INTERVAL)

    # ── Dinamica, en bloques que acaban en un fotograma guardado ─────────
    t_fin, fuente = hora_fin_trabajo()
    print(f"  fin del trabajo: {time.strftime('%Y-%m-%d %H:%M', time.localtime(t_fin))} ({fuente}); "
          f"margen {MARGEN_MIN} min", flush=True)

    paso_inicial = dyn.nsteps
    t0 = time.perf_counter()
    dur_bloque = None
    while dyn.nsteps < NSTEPS:
        if dur_bloque is not None and t_fin - time.time() < 1.5 * dur_bloque + 60 * MARGEN_MIN:
            break
        n = min(SAVE_INTERVAL - dyn.nsteps % SAVE_INTERVAL, NSTEPS - dyn.nsteps)
        tb = time.perf_counter()
        dyn.run(n)
        dur_bloque = time.perf_counter() - tb
    t1 = time.perf_counter()
    traj.close()
    fth.close()

    # ── Resumen ───────────────────────────────────────────────────────────
    hechos = dyn.nsteps - paso_inicial
    print(f"Este trabajo: pasos {paso_inicial} -> {dyn.nsteps} ({hechos} pasos, "
          f"{(t1 - t0) / 3600:.2f} h, {(t1 - t0) / max(hechos, 1):.4f} s/paso)")

    if dyn.nsteps < NSTEPS:
        print(f"Parado antes del limite de tiempo en el paso {dyn.nsteps} "
              f"(ultimo fotograma guardado). Para seguir: sbatch submit_continuar.sh")
        return

    datos = np.loadtxt(THERMO_FILE, ndmin=2)
    cols = THERMO_COLS.split()
    np.savez(FNAME + "_thermo.npz", **{k: datos[:, cols.index(c)] for k, c in
                                       [("step", "step"), ("T_K", "T_K"), ("Epot", "Epot"),
                                        ("Ekin", "Ekin"), ("Etot", "Etot"), ("Econs", "Econs")]})
    if EXPORTAR_XYZ:
        exportar_xyz()
    T_arr = datos[:, cols.index("T_K")]
    print("Nose-Hoover MD completada")
    print(f"  Pasos          : {NSTEPS}")
    print(f"  T media        : {T_arr.mean():.1f} ± {T_arr.std():.1f} K")
    print(f"  T final        : {T_arr[-1]:.1f} K")


if __name__ == "__main__":
    main()
