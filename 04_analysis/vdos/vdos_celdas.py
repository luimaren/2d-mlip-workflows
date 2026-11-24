#!/usr/bin/env python3
"""
VDOS de BGDY: celda 3x3 frente a 15x15, a 500 K y 1500 K.

Para cada temperatura se comparan 4 casos en un mismo panel:
  - 3x3,   trayectoria completa
  - 3x3,   primeros N_STEPS_CORTO pasos de MD
  - 15x15, trayectoria completa
  - 15x15, primeros N_STEPS_CORTO pasos de MD

Estructura de carpetas (nombres configurables en TEMPERATURAS y CELDAS):
    BASE_DIR/<carpeta de T>/<carpeta de celda>/<lo que sea>.traj
En cada carpeta de celda debe haber un solo archivo que cumpla TRAJ_PATRON.

Uso:
    python vdos_celdas.py                  # usa BASE_DIR
    python vdos_celdas.py /ruta/base       # otra carpeta base
    python vdos_celdas.py --replot         # rehace la figura desde vdos_celdas.npz

Memoria: las velocidades se guardan en float32. La 15x15 (3150 átomos) con
100 000 frames ocupa ~3.8 GB; pide ~16 GB. En LUMI, mejor en un nodo de cálculo:
    #!/bin/bash
    #SBATCH --account=project_XXXXXXXXX
    #SBATCH --partition=small
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=16G
    #SBATCH --time=01:00:00
    python vdos_celdas.py /ruta/base
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # sin pantalla (LUMI)
import matplotlib.pyplot as plt
from scipy import fft as sp_fft
from scipy.ndimage import gaussian_filter1d
from scipy.signal import windows
from ase.io import Trajectory


# ── Carpetas ──────────────────────────────────────────────────────────
BASE_DIR = "."

TEMPERATURAS = {          # T (K) -> carpeta de esa temperatura, relativa a BASE_DIR
    500:  "500K",
    1500: "1500K",
}
CELDAS = {                # etiqueta -> subcarpeta dentro de cada carpeta de T
    "3x3":   "3x3",
    "15x15": "15x15",
}
TRAJ_PATRON = "*.traj"    # archivo que se lee dentro de cada carpeta de celda

# ── MD: NVT con dt = 1 fs, guardada 1 de cada 2 pasos ─────────────────
TIMESTEP_FS     = 2.0     # tiempo entre frames del .traj (fs) -> eje de frecuencias
PASOS_POR_FRAME = 2       # pasos de MD entre frames guardados
N_STEPS_CORTO   = 10_000  # pasos de MD del caso "primeros N pasos"

# ── Parámetros de la VDOS (los de tu código) ──────────────────────────
THZ_TO_CM  = 33.35641     # THz → cm⁻¹
WIDTH_CM   = 10.0         # anchura suavizado gaussiano en cm⁻¹
FREQ_MAX   = 2500         # límite superior eje x en cm⁻¹
EQUIL_FRAC = 0.05         # fracción inicial descartada como equilibración (de cada ventana)

# ── Estilo y salida ───────────────────────────────────────────────────
COLORES = {"3x3": "tab:blue", "15x15": "tab:red"}
SALIDA  = "vdos_celdas"   # -> vdos_celdas.pdf, .png y .npz

# ── Rendimiento ───────────────────────────────────────────────────────
N_HILOS      = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))   # hilos de la FFT
BLOQUE_BYTES = 256 * 2**20  # tamaño del bloque de columnas en cada FFT


def buscar_traj(ruta: Path) -> Path:
    """Devuelve el .traj de una carpeta (o la propia ruta si ya es un archivo)."""
    if ruta.is_file():
        return ruta
    if not ruta.is_dir():
        raise FileNotFoundError(f"No existe la carpeta {ruta}")
    trajs = sorted(ruta.glob(TRAJ_PATRON))
    if len(trajs) != 1:
        lista = "\n    ".join(str(t) for t in trajs) or "(ninguno)"
        raise RuntimeError(
            f"En {ruta} hay {len(trajs)} archivos '{TRAJ_PATRON}' y debe haber 1:\n    {lista}\n"
            "Cambia TRAJ_PATRON o pon la ruta del archivo en TEMPERATURAS/CELDAS."
        )
    return trajs[0]


def leer_velocidades(path: Path) -> np.ndarray:
    """Velocidades de todos los frames, forma (n_frames, 3·n_átomos), float32.

    Lee solo el array de momentos de cada frame del .traj (mucho más rápido que
    construir un Atoms por frame). Si no se puede, usa traj[k].get_velocities().
    Las unidades no importan: la VDOS se normaliza al final.
    """
    traj    = Trajectory(str(path))
    nframes = len(traj)
    atoms0  = traj[0]
    natoms  = len(atoms0)
    masas   = atoms0.get_masses()[:, None]
    v0      = atoms0.get_velocities()
    if not np.any(v0):
        raise RuntimeError(f"{path}: el primer frame no tiene velocidades (momenta).")

    # Lectura rápida: comprobar que da lo mismo que ASE en el primer frame
    try:
        p0     = traj.backend[0].get("momenta")
        rapido = p0 is not None and np.allclose(p0 / masas, v0)
    except Exception:
        rapido = False

    V = np.empty((nframes, 3 * natoms), dtype=np.float32)
    print(f"  {path}: {nframes} frames, {natoms} átomos "
          f"({V.nbytes / 2**30:.2f} GB en memoria, lectura {'rápida' if rapido else 'con ASE'})")
    t0, aviso = time.time(), max(1, nframes // 10)
    for k in range(nframes):
        if rapido:
            p = traj.backend[k].get("momenta")
            if p is None:
                raise RuntimeError(f"{path}: el frame {k} no tiene momentos.")
            V[k] = (p / masas).ravel()
        else:
            V[k] = traj[k].get_velocities().ravel()
        if (k + 1) % aviso == 0:
            print(f"    {k + 1}/{nframes} frames ({time.time() - t0:.0f} s)", flush=True)
    traj.close()
    return V


def compute_vdos(V: np.ndarray, timestep_fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Mismo cálculo que el código original, sobre una matriz de velocidades ya leída.

    V: (n_frames, 3·n_átomos). La FFT se hace por bloques de columnas para no
    duplicar la memoria; el resultado es el mismo que con la matriz entera.
    """
    nframes, ndof = V.shape

    # Ventana de Hann
    window = windows.hann(nframes).astype(np.float32)[:, None]

    # VDOS promedio sobre todos los átomos y componentes: <|FFT(v·w)|²>
    potencia = np.zeros(nframes // 2 + 1)
    bloque   = max(1, int(BLOQUE_BYTES // (4 * nframes)))
    for j in range(0, ndof, bloque):
        F = sp_fft.rfft(V[:, j:j + bloque] * window, axis=0, workers=N_HILOS)
        potencia += np.sum(F.real**2 + F.imag**2, axis=1, dtype=np.float64)
    vdos = potencia / ndof / nframes

    # Eje de frecuencias
    dt_ps    = timestep_fs / 1000.0
    freq_thz = np.fft.rfftfreq(nframes, d=dt_ps)
    freq_cm  = freq_thz * THZ_TO_CM

    # Suavizado gaussiano
    df          = freq_thz[1] - freq_thz[0]
    sigma       = (WIDTH_CM / THZ_TO_CM) / df
    vdos_smooth = gaussian_filter1d(vdos, sigma=sigma)

    # Normalización al máximo
    vdos_smooth /= vdos_smooth.max()

    return freq_cm, vdos_smooth


def ventanas(n_total: int) -> dict:
    """Frames [i0, i1) de cada caso. Cada ventana descarta su EQUIL_FRAC inicial,
    como hacía el código original con la trayectoria que se le diera."""
    n_corto = N_STEPS_CORTO // PASOS_POR_FRAME
    if n_corto > n_total:
        raise RuntimeError(f"La trayectoria tiene {n_total} frames y el caso corto pide {n_corto}.")
    return {
        "completa": (int(EQUIL_FRAC * n_total), n_total),
        "corta":    (int(EQUIL_FRAC * n_corto), n_corto),
    }


def _miles(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def etiqueta(celda: str, caso: str, n_frames: int) -> str:
    if caso == "completa":
        pasos = (n_frames - 1) * PASOS_POR_FRAME      # el frame 0 es el paso 0
        return f"{celda}, full run ({_miles(pasos)} steps, {pasos * TIMESTEP_FS / PASOS_POR_FRAME / 1000:.0f} ps)"
    return f"{celda}, first {_miles(N_STEPS_CORTO)} steps ({N_STEPS_CORTO * TIMESTEP_FS / PASOS_POR_FRAME / 1000:.0f} ps)"


def calcular(base: Path) -> tuple[dict, list]:
    # Resolver todas las rutas antes de leer nada (si falta una, falla al principio)
    rutas = {}
    for T, dir_T in TEMPERATURAS.items():
        for celda, dir_c in CELDAS.items():
            rutas[(T, celda)] = buscar_traj(base / dir_T / dir_c)
    print("Trayectorias:")
    for (T, celda), p in rutas.items():
        print(f"  {T:>5} K  {celda:>6}  {p}")

    curvas, meta = {}, []
    for (T, celda), path in rutas.items():
        print(f"\n{T} K, {celda}")
        V = leer_velocidades(path)
        n_total = len(V)
        for caso, (i0, i1) in ventanas(n_total).items():
            t0 = time.time()
            freq, vdos = compute_vdos(V[i0:i1], TIMESTEP_FS)
            clave = f"T{T}_{celda}_{caso}"
            curvas[clave + "_freq"] = freq
            curvas[clave + "_vdos"] = vdos
            meta.append(dict(clave=clave, T=T, celda=celda, caso=caso, traj=str(path),
                             frames_usados=[i0, i1], label=etiqueta(celda, caso, n_total)))
            print(f"  {caso:>8}: frames {i0}–{i1} ({(i1 - i0) * TIMESTEP_FS / 1000:.1f} ps tras "
                  f"descartar {i0 * TIMESTEP_FS / 1000:.1f} ps), "
                  f"resolución {1e3 / ((i1 - i0) * TIMESTEP_FS) * THZ_TO_CM:.2f} cm⁻¹, "
                  f"{time.time() - t0:.0f} s")
        del V

    np.savez(f"{SALIDA}.npz", meta=np.array(json.dumps(meta)), **curvas)
    print(f"\nCurvas guardadas en {SALIDA}.npz")
    return curvas, meta


def dibujar(curvas: dict, meta: list) -> None:
    temps = sorted({m["T"] for m in meta})
    fig, axes = plt.subplots(len(temps), 1, figsize=(9, 4 * len(temps)),
                             sharex=True, squeeze=False)
    for ax, T in zip(axes[:, 0], temps):
        for m in (m for m in meta if m["T"] == T):
            completa = m["caso"] == "completa"
            freq, vdos = curvas[m["clave"] + "_freq"], curvas[m["clave"] + "_vdos"]
            mask = (freq >= 0) & (freq <= FREQ_MAX)
            ax.plot(freq[mask], vdos[mask],
                    label=m["label"],
                    color=COLORES.get(m["celda"]),
                    linewidth=2 if completa else 1.4,
                    linestyle="-" if completa else "--",
                    alpha=1 if completa else 0.9)
        ax.set_title(f"{T} K", fontsize=14, fontweight="bold")
        ax.set_ylabel("VDOS (normalized)", fontsize=13)
        ax.set_xlim(0, FREQ_MAX)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(fontsize=10, framealpha=0.85, loc="best")
    axes[-1, 0].set_xlabel("Frequency (cm$^{-1}$)", fontsize=13)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{SALIDA}.{ext}", dpi=150)
    print(f"Figura: {SALIDA}.pdf y {SALIDA}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="VDOS 3x3 vs 15x15 (completa y primeros pasos)")
    ap.add_argument("base", nargs="?", default=BASE_DIR, help="carpeta con las carpetas de temperatura")
    ap.add_argument("--replot", action="store_true", help=f"solo redibujar desde {SALIDA}.npz")
    args = ap.parse_args()

    if args.replot:
        datos  = np.load(f"{SALIDA}.npz")
        meta   = json.loads(str(datos["meta"]))
        curvas = {k: datos[k] for k in datos.files if k != "meta"}
    else:
        curvas, meta = calcular(Path(args.base))
    dibujar(curvas, meta)


if __name__ == "__main__":
    main()
