"""
Relajacion + MD NVT (Nose-Hoover chain) de BGDY 15x15 con el modelo MACE FTv30.
Convertido de VDOS_BGDY15x15_FTv30.ipynb. Cambia los parametros de abajo y lanza:
    sbatch submit_md.sh
"""

import os
import time

import numpy as np
import torch
from ase import units
from ase.filters import UnitCellFilter
from ase.io import Trajectory, read, write
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import BFGS, FIRE, FIRE2, LBFGSLineSearch
from mace.calculators import MACECalculator

# ---------------- Parametros ----------------
MODEL_PATH    = "FTv30-96.model"
INPUT_XYZ     = "bgdy_15x15.xyz"
TEMP          = 500        # K
TIMESTEP_FS   = 1.0
NSTEPS        = 100000
SAVE_INTERVAL = 2          # guarda en la .traj cada 2 pasos
LOG_INTERVAL  = 100        # log termodinamico cada 100 pasos
EXPORTAR_XYZ  = True       # ademas de la .traj, exporta la trayectoria a .xyz
# --------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("GPU not available!")
print(f"GPU: {torch.cuda.get_device_name(0)} | PyTorch {torch.__version__}", flush=True)


def NoseHooverMD(
    init_conf,
    temp,
    calc,
    fname,
    save_interval,
    nsteps,
    init_temp=None,
    timestep_fs=1.0,
    tdamp_fs=100.0,
    log_interval=100,
):
    conf = init_conf.copy()
    conf.calc = calc

    if init_temp is None:
        init_temp = temp
    if abs(init_temp - temp) > 200:
        print(f" Delta_T = {abs(init_temp - temp):.0f} K — considera equilibración larga")

    # ── Inicialización térmica ────────────────────────────────────────
    MaxwellBoltzmannDistribution(conf, temperature_K=init_temp)
    Stationary(conf)
    ZeroRotation(conf)

    # ── Limpiar archivos antiguos ─────────────────────────────────────
    for ext in [".traj", ".xyz", "_thermo.npz"]:
        if os.path.exists(fname + ext):
            os.remove(fname + ext)

    traj = Trajectory(fname + ".traj", "w", conf)

    dyn = NoseHooverChainNVT(
        atoms=conf,
        timestep=timestep_fs * units.fs,
        temperature_K=temp,
        tdamp=tdamp_fs * units.fs,
    )

    # ── Callbacks ────────────────────────────────────────────────────
    dyn.attach(traj.write, interval=save_interval)

    # Log termodinámico
    log_data = {"step": [], "T_K": [], "Epot": [], "Ekin": [], "Etot": []}

    def log_thermo():
        log_data["step"].append(dyn.get_number_of_steps())
        log_data["T_K"].append(conf.get_temperature())
        epot = conf.get_potential_energy() / len(conf)
        ekin = conf.get_kinetic_energy() / len(conf)
        log_data["Epot"].append(epot)
        log_data["Ekin"].append(ekin)
        log_data["Etot"].append(epot + ekin)
        # Progreso en el .out del trabajo
        print(f"paso {log_data['step'][-1]:>7d}/{nsteps}  T = {log_data['T_K'][-1]:7.1f} K", flush=True)

    dyn.attach(log_thermo, interval=log_interval)

    # ── Dinámica ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    dyn.run(nsteps)
    t1 = time.perf_counter()
    traj.close()

    # ── Exportar XYZ ──────────────────────────────────────────────────
    if EXPORTAR_XYZ:
        write(fname + ".xyz", Trajectory(fname + ".traj"))

    # ── Guardar log termodinámico ─────────────────────────────────────
    np.savez(fname + "_thermo.npz", **{k: np.array(v) for k, v in log_data.items()})

    # ── Resumen ───────────────────────────────────────────────────────
    total_time    = t1 - t0
    time_per_step = total_time / nsteps
    T_arr         = np.array(log_data["T_K"])
    print("Nose–Hoover MD completada")
    print(f"  Pasos          : {nsteps}")
    print(f"  Tiempo total   : {total_time:.2f} s")
    print(f"  Tiempo/paso    : {time_per_step:.4f} s/step")
    print(f"  T media        : {T_arr.mean():.1f} ± {T_arr.std():.1f} K")
    print(f"  T final        : {T_arr[-1]:.1f} K")


def relax_structure(atoms, calculator, optimizer="BFGS",
                    fmax=0.002, output_file="relax", steps=10000,
                    relax_cell=False):
    """
    Relaja una estructura con distintos optimizadores de ASE.
    Si relax_cell=True, también se relaja la celda (volumen y forma).
    """
    def remove_if_exists(*filenames):
        for fname in filenames:
            if os.path.isfile(fname):
                os.remove(fname)

    atoms = atoms.copy()
    atoms.calc = calculator

    optimizers = {"BFGS": BFGS, "LBFGSLineSearch": LBFGSLineSearch, "FIRE": FIRE, "FIRE2": FIRE2}
    if optimizer not in optimizers:
        raise ValueError(f"Optimizador no soportado: {optimizer}")

    suffix = "_cell" if relax_cell else ""
    traj_file     = f"{output_file}_{optimizer}{suffix}.traj"
    log_file      = f"{output_file}_{optimizer}{suffix}.log"
    xyz_file      = f"{output_file}_{optimizer}{suffix}.xyz"
    full_xyz_file = f"{output_file}_{optimizer}{suffix}_full.xyz"
    remove_if_exists(traj_file, log_file, xyz_file, full_xyz_file)

    system = UnitCellFilter(atoms) if relax_cell else atoms
    opt = optimizers[optimizer](system, trajectory=traj_file, logfile=log_file)
    opt.run(fmax=fmax, steps=steps)

    write(xyz_file, atoms)
    write(full_xyz_file, read(traj_file, ":"))
    return atoms


# ---------------- Ejecucion ----------------
FTv30 = MACECalculator(model_paths=MODEL_PATH, device="cuda", default_dtype="float64")

# Si ya existe la estructura relajada (p. ej. copiada de otra carpeta), se reutiliza
if os.path.isfile("relax_bgdy_15x15_BFGS.xyz"):
    relax_bgdy_15x15 = read("relax_bgdy_15x15_BFGS.xyz")
else:
    bgdy_15x15 = read(INPUT_XYZ)
    relax_bgdy_15x15 = relax_structure(bgdy_15x15, FTv30, optimizer="BFGS",
                                       output_file="relax_bgdy_15x15", relax_cell=False)

NoseHooverMD(init_conf=relax_bgdy_15x15, temp=TEMP, calc=FTv30,
             fname=f"md_bgdy_15x15_{TEMP}_{TIMESTEP_FS:g}fs",
             save_interval=SAVE_INTERVAL, timestep_fs=TIMESTEP_FS,
             nsteps=NSTEPS, log_interval=LOG_INTERVAL)
