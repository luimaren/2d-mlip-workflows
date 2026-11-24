#!/usr/bin/env python
"""Rampa térmica de BGDY 3x3 con MACE-MH-1 y Langevin NPT en el plano: versión para LUMI-G (SLURM).

Mismo código que el cuaderno rampa_bgdy.ipynb (relajación, equilibrado, rampa, seguimiento y figuras),
sin hilos: cada modo corre en primer plano y escribe en la misma carpeta resultados/.

Modos:
  python rampa_npt_bgdy.py [todo]        hace lo que falte: relajación → equilibrado → rampa → figuras.
                                         Si algo está a medias, lo reanuda desde su último frame.
  python rampa_npt_bgdy.py prueba        relajación + 1 ps de equilibrado + 5 ps de rampa en
                                         resultados_prueba/: da ms/paso y las horas de la rampa completa.
  python rampa_npt_bgdy.py relajacion | equilibrado | rampa  [--reanudar]
  python rampa_npt_bgdy.py figuras [--semillas 1 2 3]       figuras y análisis de lo que haya
  python rampa_npt_bgdy.py estado                           últimas líneas de progreso (sin GPU)

Parada limpia: con SIGUSR1 o SIGTERM (SLURM los manda antes del límite de tiempo con --signal) termina el
bloque en curso (≤ 20 fs), guarda un frame y sale con código 3 para que el trabajo se reenvíe.
Códigos de salida: 0 completa, 3 interrumpida (hay que reanudar), 4 parada por colapso del MLIP, 1 error.
"""
import argparse
import shutil
import signal
import socket

import matplotlib
matplotlib.use("Agg")                    # sin pantalla: las figuras van a PNG

import json, os, re, sys, threading, time, warnings

import numpy as np
import matplotlib.pyplot as plt
from ase import units
from ase.filters import FrechetCellFilter
from ase.geometry import find_mic
from ase.io import read, write
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS
from scipy.signal import find_peaks
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

warnings.filterwarnings("ignore", message=".*weights_only.*")        # aviso de torch.load en e3nn y mace


# ==============================================================================================
# Parámetros
# ==============================================================================================
# --- Archivos y modelo ---
ESTRUCTURA = "bgdy_3x3.xyz"      # C108B18, pbc = T T F, vacío de 20 Å
MODELO = "FT_MACE_v32_c4.model"
HEAD = "oc20_usemppbe"           # cabeza de MH-1
DEVICE = "cuda"
DTYPE_MD = "float32"             # la dinámica en float32 (~2x más rápida); la relajación siempre en float64
CUEQ = False                     # True: acelera con cuEquivariance si está instalado (se comprueba que da
                                 # la misma energía y fuerzas que el modelo normal)
RESULTADOS = "resultados"        # carpeta raíz de todas las salidas
SEMILLA = int(os.environ.get("SEMILLA", 1))   # velocidades y ruido; otra réplica: sbatch --export=ALL,SEMILLA=2

# --- 1. Relajación (posiciones + celda en el plano, c fijo) ---
FMAX = 1e-3                      # eV/Å
PASOS_MAX_BFGS = 2000
DEFORMACION_K2D = 0.005          # deformación para estimar el módulo 2D (masa del pistón)

# --- Dinámica de Langevin NPT (común a equilibrado y rampa) ---
DT_FS = 1.0                      # fs (con 2 fs la energía deriva por encima de ~3000 K)
FRICCION = 1.0                   # ps^-1, termostato de los átomos
TENSION_EXT = 0.0                # N/m, tensión externa en el plano (0 = presión cero)
PERIODO_PISTON = 1.0             # ps, periodo de oscilación del área (fija la masa del pistón)
FRICCION_PISTON = 2.0            # ps^-1, termostato del pistón
AREA_MAX = 1.5                   # si el área supera 1.5 veces la de 0 K (red rota), el área se congela
D_MIN_PARADA = 0.8               # Å: si dos átomos se acercan más, el MLIP ha colapsado y se para
LOG_FS = 20.0                    # fs entre filas de log.csv
FRAME_PS = 1.0                   # ps entre frames (posiciones, momentos, celda; sirven para reanudar)

# --- 2. Equilibrado ---
T_EQUIL = 500.0                 # K
EQUIL_PS = 10.0                  # ps
FRICCION_INICIAL = 10.0          # ps^-1 durante los primeros TERMALIZACION_PS (calienta en ~0.3 ps)
TERMALIZACION_PS = 2.0
PROGRESO_EQUIL_PS = 1.0          # ps entre líneas de progreso

# --- 3. Rampa ---
T_INI = 500.0                   # K
T_FIN = 3500.0                   # K
RATE = 2.0                       # K/ps (= 2·10³ K/ns = 2·10¹² K/s): 1000 ps, 10⁶ pasos
PROGRESO_RAMPA_PS = 10.0         # ps entre líneas de progreso


def carpeta_equilibrado():
    return f"{RESULTADOS}/02_equilibrado_{T_EQUIL:.0f}K_s{SEMILLA}"


def carpeta_rampa(semilla=None):
    s = SEMILLA if semilla is None else semilla
    return f"{RESULTADOS}/03_rampa_{T_INI:.0f}-{T_FIN:.0f}K_{RATE:g}Kps_s{s}"




# --- Prueba corta (modo prueba) ---
PRUEBA_EQUIL_PS = 1.0            # ps de equilibrado en la prueba
PRUEBA_RAMPA_PS = 5.0            # ps de rampa en la prueba


# ==============================================================================================
# Modelo e integrador de Langevin NPT para láminas
# ==============================================================================================
NM = 1 / 16.0218             # 1 N/m en eV/Å²
PS = 1000 * units.fs         # 1 ps en unidades de tiempo de ASE
PBC_LAMINA = [True, True, False]
VACIO_MIN_PBC = 40.0         # Å: c mínimo si hay que usar pbc = True (mace-torch < 0.3.17)


# ----------------------------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------------------------
def titulo(texto):
    print(f"\n{'=' * 100}\n{texto}\n{'=' * 100}", flush=True)


def carpeta(ruta):
    os.makedirs(ruta, exist_ok=True)
    return ruta


def guardar_json(ruta, datos):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False, default=float)


def leer_json(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def area(atoms):
    """Área de la celda en el plano (Å²)."""
    return abs(np.linalg.det(atoms.cell.array[:2, :2]))


def media_error(x, n_bloques=10):
    """Media y error típico por bloques (tiene en cuenta la correlación temporal)."""
    x = np.asarray(x)
    medias = [b.mean() for b in np.array_split(x, n_bloques) if len(b)]
    return x.mean(), np.std(medias, ddof=1) / np.sqrt(len(medias))


def masa_piston(K2D, area_ref, periodo_ps):
    """Masa del pistón (eV·t²) para que el área oscile con el periodo dado.

    Para una deformación isótropa en el plano, A = A_ref·exp(2ε), la energía elástica es
    2·K2D·A·ε², así que ω² = 4·K2D·A/W.
    """
    return 4 * K2D * NM * area_ref * (periodo_ps * PS / (2 * np.pi)) ** 2


# ----------------------------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------------------------
def cargar_modelo(dtype, comprobar_en=None):
    """MACECalculator con la cabecera HEAD. Con CUEQ = True usa cuEquivariance, solo si da lo mismo que
    e3nn (cuEquivariance es solo para GPU NVIDIA: en LUMI, CUEQ = False)."""
    from mace.calculators import MACECalculator

    kw = dict(model_paths=MODELO, head=HEAD, device=DEVICE, default_dtype=dtype)
    calc = MACECalculator(**kw)
    if not CUEQ:
        return calc
    try:
        rapido = MACECalculator(**kw, enable_cueq=True)
    except Exception as e:                                              # noqa: BLE001
        print(f"⚠ cuEquivariance no disponible ({type(e).__name__}: {e}). Se usa el modelo normal.")
        return calc
    if comprobar_en is not None:
        a1, a2 = comprobar_en.copy(), comprobar_en.copy()
        a1.calc, a2.calc = calc, rapido
        dE = abs(a1.get_potential_energy() - a2.get_potential_energy()) / len(a1)
        dF = np.abs(a1.get_forces() - a2.get_forces()).max()
        if dE > 1e-4 or dF > 5e-3:
            print(f"⚠ cuEquivariance difiere (ΔE = {dE:.1e} eV/át, ΔF = {dF:.1e} eV/Å). Se usa el modelo normal.")
            return calc
        print(f"cuEquivariance activo (ΔE = {dE:.1e} eV/át, ΔF = {dF:.1e} eV/Å respecto a e3nn).")
    return rapido


# ----------------------------------------------------------------------------------------------
# Integrador
# ----------------------------------------------------------------------------------------------
class LangevinNPT2D:
    """Dinámica de Langevin NPT para una lámina 2D (periódica en x, y; z libre; c fijo).

    - Átomos: Langevin con el esquema BAOAB (Leimkuhler y Matthews, JCP 138, 174102, 2013).
    - Celda: pistón de Langevin isótropo en el plano (a y b se escalan igual; c, γ y el vacío no
      cambian). Ecuaciones de Martyna, Tobias y Klein (JCP 101, 4177, 1994) para d = 2, con un
      termostato de Langevin sobre el pistón (Quigley y Probert, JCP 120, 11432, 2004).

    Con A = A_ref·exp(2ε), momento p del pistón, masa W, N_f = 3N − 3 y tensión externa τ_ext:
        dr_α/dt  = v_α + (p/W)·r_α                                  (α = x, y)
        m·dv/dt  = F − (δ_α + 2/N_f)·(p/W)·m·v − γ·m·v + ruido      (δ_α = 1 en x, y; 0 en z)
        dp/dt    = Σ m(v_x² + v_y²) − V(σ_xx + σ_yy) + 2·A·τ_ext + (2/N_f)·Σ m·v² − γ_p·p + ruido
    Sin termostatos (γ = γ_p = 0) conserva E_tot + p²/2W − τ_ext·A: sirve para comprobar la
    implementación. Con ellos muestrea el colectivo NPT (área con sus fluctuaciones correctas).

    Se fija el centro de masas (se quita su velocidad tras el ruido): T = Σ m·v² / (N_f·k_B).
    Una sola evaluación del modelo (fuerzas + stress) por paso.
    """

    def __init__(self, atoms, dt_fs, temperatura, friccion, masa_piston, friccion_piston,
                 tension_ext=0.0, barostato=True, p_eps=0.0, semilla=None):
        self.atoms = atoms
        self.dt = dt_fs * units.fs
        self.m = atoms.get_masses()[:, None]
        self.M = self.m.sum()
        self.nf = 3 * len(atoms) - 3
        self.W = masa_piston
        self.tension_ext = tension_ext * NM                         # eV/Å²
        self.barostato = barostato
        self.p_eps = float(p_eps) if barostato else 0.0
        self.rng = np.random.default_rng(semilla)
        self.v = atoms.get_velocities()
        self.v -= (self.m * self.v).sum(0) / self.M
        self.kT = units.kB * temperatura
        self.gamma = friccion / PS
        self.gamma_p = friccion_piston / PS
        self._coeficientes()
        self._fuerzas()

    # --- parámetros que pueden cambiar durante la dinámica ---
    def set_temperature(self, temperatura):
        self.kT = units.kB * temperatura
        self._coeficientes()

    def set_friction(self, friccion):
        self.gamma = friccion / PS
        self._coeficientes()

    def congelar_area(self):
        self.barostato, self.p_eps = False, 0.0

    def _coeficientes(self):
        self.c1 = np.exp(-self.gamma * self.dt)
        self.c2 = np.sqrt(self.kT * (1 - self.c1 ** 2) / self.m)
        self.cp1 = np.exp(-self.gamma_p * self.dt)
        self.cp2 = np.sqrt(self.W * self.kT * (1 - self.cp1 ** 2))

    # --- fuerzas ---
    def _fuerzas(self):
        self.a = self.atoms.get_forces() / self.m
        if self.barostato:
            s = self.atoms.get_stress(voigt=True)                    # eV/Å³, sin término cinético
            self.F_conf = -self.atoms.get_volume() * (s[0] + s[1]) + 2 * area(self.atoms) * self.tension_ext

    def _F_piston(self):
        mv2 = self.m * self.v ** 2
        return self.F_conf + mv2[:, :2].sum() + 2 / self.nf * mv2.sum()

    def _escalar_plano(self, s):
        celda = self.atoms.cell.array.copy()
        celda[:2, :2] *= s
        self.atoms.set_cell(celda, scale_atoms=False)
        self.atoms.positions[:, :2] *= s

    # --- un paso BAOAB ---
    def step(self):
        h = self.dt
        if self.barostato:
            self.p_eps += 0.5 * h * self._F_piston()                 # B
        self.v += 0.5 * h * self.a
        if self.barostato:
            self._escalar_plano(np.exp(0.5 * h * self.p_eps / self.W))  # A
        self.atoms.positions += 0.5 * h * self.v
        if self.barostato:                                           # O: arrastre del pistón + ruido
            x = self.p_eps / self.W * h
            self.v[:, :2] *= np.exp(-(1 + 2 / self.nf) * x)
            self.v[:, 2] *= np.exp(-(2 / self.nf) * x)
            self.p_eps = self.cp1 * self.p_eps + self.cp2 * self.rng.standard_normal()
        self.v = self.c1 * self.v + self.c2 * self.rng.standard_normal(self.v.shape)
        self.v -= (self.m * self.v).sum(0) / self.M
        self.atoms.positions += 0.5 * h * self.v                     # A
        if self.barostato:
            self._escalar_plano(np.exp(0.5 * h * self.p_eps / self.W))
        self._fuerzas()                                              # modelo: fuerzas + stress
        self.v += 0.5 * h * self.a                                   # B
        if self.barostato:
            self.p_eps += 0.5 * h * self._F_piston()
        self.atoms.set_velocities(self.v)

    # --- magnitudes ---
    def temperatura(self):
        return (self.m * self.v ** 2).sum() / (self.nf * units.kB)

    def energia_conservada(self):
        """E_tot + p²/2W − τ_ext·A (constante sin termostatos)."""
        return (self.atoms.get_potential_energy() + 0.5 * (self.m * self.v ** 2).sum()
                + 0.5 * self.p_eps ** 2 / self.W - self.tension_ext * area(self.atoms))



def version_mace():
    import mace
    return mace.__version__, tuple(int(n) for n in re.findall(r"\d+", mace.__version__)[:3])


def ampliar_vacio(atoms, c_min):
    """c ≥ c_min con la lámina centrada en z (para pbc = True)."""
    celda = atoms.cell.array.copy()
    celda[2, 2] = max(celda[2, 2], c_min)
    atoms.set_cell(celda, scale_atoms=False)
    atoms.positions[:, 2] += celda[2, 2] / 2 - atoms.positions[:, 2].mean()
    return atoms


def pbc_modelo():
    """pbc de las etapas de dinámica, el que decidió la relajación."""
    r = leer_json(f"{RESULTADOS}/01_relajacion/resumen.json")
    return PBC_LAMINA if r.get("pbc_modelo", "T T F") == "T T F" else True


PBC_MODELO = PBC_LAMINA      # lo decide la relajación (T T T si mace-torch < 0.3.17)



# ----------------------------------------------------------------------------------------------
# Análisis estructural
# ----------------------------------------------------------------------------------------------
R_ENLACE = 1.85      # Å, vecinos para la coordinación y la conectividad
R_ROTURA = 2.2       # Å, un B–C de la red inicial cuenta como roto por encima
FRAG_PEQ = 3         # átomos: fragmentos de este tamaño o menos cuentan como sueltos


class Referencia:
    """Enlaces y coordinaciones de la lámina relajada, para comparar cada frame con ella."""

    def __init__(self, ref):
        sim = np.array(ref.get_chemical_symbols())
        self.esB = sim == "B"
        self.z_ideal = np.where(self.esB, 3, 2)                       # B con 3 C; C con 2 vecinos
        i, j = neighbor_list("ij", ref, R_ENLACE)
        self.BC = np.column_stack([i, j])[self.esB[i] & (sim[j] == "C")]
        self.n = len(ref)
        self.area = area(ref)
        if not (np.bincount(i, minlength=self.n) == self.z_ideal).all():
            print("⚠ La referencia no tiene la coordinación ideal (B 3, C 2): revisa la estructura.")

    def analizar(self, atoms):
        i, j, d = neighbor_list("ijd", atoms, R_ENLACE)
        z = np.bincount(i, minlength=self.n)
        _, dBC = find_mic(atoms.positions[self.BC[:, 1]] - atoms.positions[self.BC[:, 0]],
                          atoms.cell, atoms.pbc)
        n_frag, etiqueta = connected_components(
            coo_matrix((np.ones(len(i)), (i, j)), shape=(self.n, self.n)), directed=False)
        tam = np.bincount(etiqueta)
        mayor = etiqueta == tam.argmax()
        return {"BC_intactos": (dBC < R_ROTURA).mean(), "coord_ideal": (z == self.z_ideal).mean(),
                "red_mayor": mayor.mean(), "sueltos": (tam[etiqueta] <= FRAG_PEQ).mean(),
                "n_fragmentos": n_frag, "h_rms_A": atoms.positions[mayor, 2].std(),
                "d_min_A": d.min() if len(d) else np.nan}


def longitudes_enlace(atoms):
    """Longitudes medias de B–C, C≡C y C–C (simple) en Å."""
    sim = np.array(atoms.get_chemical_symbols())
    i, j, d = neighbor_list("ijd", atoms, R_ENLACE)
    m = i < j
    i, j, d = i[m], j[m], d[m]
    bc = (sim[i] != sim[j])
    cc = (sim[i] == "C") & (sim[j] == "C")
    media = lambda x: float(x.mean()) if len(x) else float("nan")
    return {"B-C": media(d[bc]), "C≡C": media(d[cc & (d < 1.28)]), "C-C": media(d[cc & (d >= 1.28)]),
            "n_BC": int(bc.sum()), "n_CtripleC": int((cc & (d < 1.28)).sum()), "n_CC": int((cc & (d >= 1.28)).sum())}


# ----------------------------------------------------------------------------------------------
# Seguimiento: log.csv (cada LOG_FS), estructura.csv y frames.extxyz (cada FRAME_PS), progreso.txt
# ----------------------------------------------------------------------------------------------
COLUMNAS_LOG = ["t_ps", "T_obj_K", "T_K", "T_xy_K", "T_z_K", "T_B_K", "T_C_K", "Epot_eV_at", "Ekin_eV_at",
                "Etot_eV_at", "H_eV_at", "area_A2_at", "a_A", "tension_Nm", "z_rms_A", "d_min_A", "barostato"]
COLUMNAS_EST = ["t_ps", "T_obj_K", "BC_intactos", "coord_ideal", "red_mayor", "sueltos", "n_fragmentos",
                "h_rms_A", "d_min_A"]


class Seguimiento:
    def __init__(self, ruta, ref, reanudar_en=None, eco=True):
        self.ruta = carpeta(ruta)
        self.eco = eco
        self.ref = ref
        self.f_log, self.f_est = f"{ruta}/log.csv", f"{ruta}/estructura.csv"
        self.f_frames, self.f_prog = f"{ruta}/frames.extxyz", f"{ruta}/progreso.txt"
        if reanudar_en is None:
            if os.path.exists(self.f_frames):
                raise RuntimeError(f"{ruta} ya tiene una dinámica: reanuda (reanudar=True) para seguirla, "
                                   "o borra la carpeta (o cambia SEMILLA) para empezar de cero.")
            for f, cols in [(self.f_log, COLUMNAS_LOG), (self.f_est, COLUMNAS_EST)]:
                with open(f, "w", encoding="utf-8") as h:
                    h.write(",".join(cols) + "\n")
            open(self.f_prog, "w", encoding="utf-8").close()
        else:                                                         # recorta al instante del frame
            for f in (self.f_log, self.f_est):
                with open(f, encoding="utf-8") as h:
                    lineas = h.readlines()
                with open(f, "w", encoding="utf-8") as h:
                    h.write(lineas[0] + "".join(l for l in lineas[1:]
                                                if float(l.split(",", 1)[0]) <= reanudar_en + 1e-6))
        self.log = open(self.f_log, "a", encoding="utf-8")
        self.est = open(self.f_est, "a", encoding="utf-8")
        self.prog = open(self.f_prog, "a", encoding="utf-8")
        self.ventana = []
        self.ultima_est = None
        self.t_ultimo_frame = -np.inf if reanudar_en is None else reanudar_en

    def mensaje(self, texto):
        if self.eco:
            print(texto, flush=True)
        self.prog.write(texto + "\n")
        self.prog.flush()

    def fila(self, t, T_obj, integ):
        a, n = integ.atoms, len(integ.atoms)
        kB = units.kB
        mv2 = integ.m * integ.v ** 2
        K = mv2.sum()
        K_B, K_C = mv2[self.ref.esB].sum(), mv2[~self.ref.esB].sum()
        epot = a.get_potential_energy() / n
        ekin = 0.5 * K / n
        A = area(a)
        s = a.get_stress(voigt=True)
        c = a.cell[2, 2]
        tension = c * (0.5 * (s[0] + s[1]) - 0.5 * mv2[:, :2].sum() / a.get_volume()) / NM   # N/m
        d = neighbor_list("d", a, 1.5)
        f = {"t_ps": t, "T_obj_K": T_obj, "T_K": K / (integ.nf * kB),
             "T_xy_K": mv2[:, :2].sum() / ((2 * n - 2) * kB), "T_z_K": mv2[:, 2].sum() / ((n - 1) * kB),
             "T_B_K": K_B / (3 * self.ref.esB.sum() * kB), "T_C_K": K_C / (3 * (~self.ref.esB).sum() * kB),
             "Epot_eV_at": epot, "Ekin_eV_at": ekin, "Etot_eV_at": epot + ekin,
             "H_eV_at": epot + ekin - integ.tension_ext * A / n, "area_A2_at": A / n,
             "a_A": np.linalg.norm(a.cell[0]), "tension_Nm": tension, "z_rms_A": a.positions[:, 2].std(),
             "d_min_A": d.min() if len(d) else 1.5, "barostato": int(integ.barostato)}
        self.log.write(",".join(f"{f[k]:.6g}" if k not in ("t_ps", "Epot_eV_at", "Etot_eV_at", "H_eV_at")
                                else f"{f[k]:.8g}" for k in COLUMNAS_LOG) + "\n")
        self.ventana.append(f)
        return f

    def frame(self, t, T_obj, integ):
        a = integ.atoms.copy()
        a.set_momenta(integ.m * integ.v)
        a.info.update(t_ps=t, T_obj_K=T_obj, p_eps=integ.p_eps, barostato=int(integ.barostato),
                      Epot_eV_at=integ.atoms.get_potential_energy() / len(a))
        write(self.f_frames, a, format="extxyz", append=True)
        e = self.ref.analizar(a)
        self.est.write(",".join(f"{v:.6g}" for v in [t, T_obj] + [e[k] for k in COLUMNAS_EST[2:]]) + "\n")
        self.ultima_est = e
        self.t_ultimo_frame = t
        self.log.flush(); self.est.flush()
        return a

    def progreso(self, t, T_obj, ms_paso, horas):
        w = self.ventana
        T = np.mean([f["T_K"] for f in w])
        e = self.ultima_est or {}
        self.mensaje(
            f"t {t:8.2f} ps | T_obj {T_obj:6.0f} | <T> {T:6.0f} K | E_pot {np.mean([f['Epot_eV_at'] for f in w]):.4f} eV/át"
            f" | área {100 * (w[-1]['area_A2_at'] * self.ref.n / self.ref.area - 1):+6.2f} %"
            f" | τ {np.mean([f['tension_Nm'] for f in w]):+6.2f} N/m | z_rms {w[-1]['z_rms_A']:4.2f} Å"
            f" | B–C {e.get('BC_intactos', np.nan):5.1%} | red {e.get('red_mayor', np.nan):5.1%}"
            f" | d_min {min(f['d_min_A'] for f in w):4.2f} Å | {ms_paso:5.0f} ms/paso, quedan {horas:5.1f} h")
        self.ventana = []

    def cerrar(self):
        for h in (self.log, self.est, self.prog):
            h.close()


# ----------------------------------------------------------------------------------------------
# Bucle de dinámica
# ----------------------------------------------------------------------------------------------
def ejecutar_md(integ, seg, t_ini, t_fin, consigna, progreso_ps, friccion=None, parar=None):
    """Integra de t_ini a t_fin (ps). consigna(t) da la T objetivo; friccion(t), la fricción (ps^-1).
    Escribe una fila de log cada LOG_FS, un frame cada FRAME_PS y una línea de progreso cada progreso_ps.
    Se para si dos átomos se acercan a menos de D_MIN_PARADA; congela el área si supera AREA_MAX.
    Si se interrumpe (botón ■, o parar.set() desde detener()), devuelve "interrumpida".
    Devuelve (estado, t final)."""
    dt_ps = DT_FS / 1000
    n_log = max(1, round(LOG_FS / DT_FS))
    logs_frame = max(1, round(FRAME_PS * 1000 / LOG_FS))
    logs_prog = max(1, round(progreso_ps * 1000 / LOG_FS))
    paso, paso_fin = round(t_ini / dt_ps), round(t_fin / dt_ps)
    d_min_parada, area_max = D_MIN_PARADA, AREA_MAX
    paso0, k, t_cpu = paso, 0, time.time()
    if paso == 0:
        seg.fila(0.0, consigna(0.0), integ)
        seg.frame(0.0, consigna(0.0), integ)
        seg.ventana = []
    try:
        while paso < paso_fin:
            if parar is not None and parar.is_set():
                t = paso * dt_ps
                if t > seg.t_ultimo_frame + 1e-9:
                    seg.frame(t, consigna(t), integ)
                seg.mensaje(f"Parada pedida en t = {t:.2f} ps: frame guardado. Para seguir, reanuda.")
                return "interrumpida", t
            n = min(n_log, paso_fin - paso)
            t_medio = (paso + n / 2) * dt_ps
            integ.set_temperature(consigna(t_medio))
            if friccion is not None:
                integ.set_friction(friccion(t_medio))
            for _ in range(n):
                integ.step()
            paso += n
            k += 1
            t = paso * dt_ps
            f = seg.fila(t, consigna(t), integ)
            if f["d_min_A"] < d_min_parada:
                seg.frame(t, consigna(t), integ)
                seg.mensaje(f"⚠ Parada en t = {t:.2f} ps (T objetivo {consigna(t):.0f} K): dos átomos a "
                            f"{f['d_min_A']:.2f} Å. El MLIP ha colapsado; lo que siguiera no sería físico.")
                return "parada por colapso", t
            if integ.barostato and f["area_A2_at"] * len(integ.atoms) > area_max * seg.ref.area:
                integ.congelar_area()
                seg.mensaje(f"⚠ t = {t:.2f} ps: el área supera {area_max:g} veces la de 0 K (la red ya no "
                            "soporta tensión). Desde aquí el área queda fija (NVT).")
            if k % logs_frame == 0 or paso >= paso_fin:
                seg.frame(t, consigna(t), integ)
            if k % logs_prog == 0 or paso >= paso_fin:
                ms = (time.time() - t_cpu) / (paso - paso0) * 1000
                seg.progreso(t, consigna(t), ms, ms * (paso_fin - paso) / 3.6e6)
    except KeyboardInterrupt:
        seg.mensaje(f"Interrumpida hacia t = {paso * dt_ps:.2f} ps. Lo guardado llega hasta el último frame; "
                    "para seguir, reanuda (reanudar=True).")
        return "interrumpida", paso * dt_ps
    return "completa", paso * dt_ps


def velocidades_iniciales(atoms, temperatura, semilla):
    """Maxwell–Boltzmann a la temperatura dada, sin velocidad del centro de masas."""
    from ase.md.velocitydistribution import Stationary
    try:
        from ase.md.velocitydistribution import thermalize_momenta            # ASE >= 3.29
        thermalize_momenta(atoms, temperatura, rng=np.random.default_rng(semilla))
    except ImportError:
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperatura, rng=np.random.default_rng(semilla))
    Stationary(atoms)


def leer_estado(ruta_frame):
    """Último frame de un fichero extxyz (posiciones, momentos, celda, p_eps, t)."""
    fr = read(ruta_frame, index=-1)
    fr.pbc = PBC_LAMINA
    return fr


def estado_en(atoms, fr):
    """Copia posiciones, celda y momentos del frame en atoms."""
    atoms.set_cell(fr.cell, scale_atoms=False)
    atoms.set_positions(fr.positions)
    atoms.set_momenta(fr.get_momenta())


def preparar_dinamica(salida, reanudar, temperatura, friccion, inicio=None, eco=True):
    """Modelo, integrador y seguimiento para una etapa de dinámica.

    Estado inicial, por orden de prioridad:
      - reanudar: el último frame de la propia etapa (posiciones, momentos, celda, pistón, t);
      - inicio: un extxyz con momentos (el final de la etapa anterior);
      - si no: la lámina relajada con velocidades de Maxwell–Boltzmann a la temperatura dada.
    La lámina relajada (etapa 1) es la referencia del área y de los enlaces, y su K2D fija la masa del pistón.
    """
    rel = f"{RESULTADOS}/01_relajacion"
    if not os.path.exists(f"{rel}/relajada.extxyz"):
        raise FileNotFoundError("Falta la lámina relajada: ejecuta antes la etapa 1.")
    ref = read(f"{rel}/relajada.extxyz")
    ref.pbc = pbc_modelo()
    K2D = leer_json(f"{rel}/resumen.json")["K2D_Nm"]
    W = masa_piston(K2D, area(ref), PERIODO_PISTON)

    atoms = ref.copy()
    atoms.calc = cargar_modelo(DTYPE_MD, comprobar_en=ref)
    pbc = ref.pbc.copy()
    t0, p_eps, baro, semilla = 0.0, 0.0, True, SEMILLA
    if reanudar:
        fr = leer_estado(f"{salida}/frames.extxyz")
        estado_en(atoms, fr)
        atoms.pbc = pbc
        t0, p_eps, baro = float(fr.info["t_ps"]), float(fr.info["p_eps"]), bool(fr.info["barostato"])
        semilla = [SEMILLA, round(t0 * 1000)]                       # otra secuencia de ruido al reanudar
        info = f"Reanudando en t = {t0:g} ps desde el último frame de {salida}."
    elif inicio is not None:
        if not os.path.exists(inicio):
            raise FileNotFoundError(f"Falta {inicio}: ejecuta antes la etapa anterior.")
        fr = leer_estado(inicio)
        estado_en(atoms, fr)
        atoms.pbc = pbc
        p_eps, baro = float(fr.info["p_eps"]), bool(fr.info["barostato"])
        semilla = [SEMILLA, 7]
        info = (f"Estado inicial: {inicio} (T = {fr.info.get('T_obj_K', float('nan')):.0f} K, "
                f"área {100 * (area(fr) / area(ref) - 1):+.2f} % respecto a 0 K).")
    else:
        velocidades_iniciales(atoms, temperatura, SEMILLA)
        info = f"Estado inicial: lámina relajada con velocidades de Maxwell–Boltzmann a {temperatura:g} K."

    integ = LangevinNPT2D(atoms, DT_FS, temperatura, friccion, W, FRICCION_PISTON,
                          TENSION_EXT, baro, p_eps, semilla)
    seg = Seguimiento(salida, Referencia(ref), reanudar_en=t0 if reanudar else None, eco=eco)
    seg.mensaje(info)
    seg.mensaje(f"Langevin NPT 2D: dt = {DT_FS:g} fs | fricción {friccion:g} ps⁻¹ | pistón isótropo en el plano: "
                f"K2D = {K2D:.1f} N/m, periodo {PERIODO_PISTON:g} ps, fricción {FRICCION_PISTON:g} ps⁻¹ | "
                f"τ_ext = {TENSION_EXT:g} N/m | {DTYPE_MD} | barostato {'activo' if baro else 'congelado'}")
    return integ, seg, t0, ref


def copiar_final(salida):
    """final.extxyz = último frame de la etapa (entrada de la siguiente)."""
    fr = leer_estado(f"{salida}/frames.extxyz")
    write(f"{salida}/final.extxyz", fr, format="extxyz")
    return fr


def leer_csv(ruta):
    """CSV con cabecera; ignora una última línea a medio escribir (rampa en marcha)."""
    with open(ruta, encoding="utf-8") as f:
        lineas = f.readlines()
    if lineas and not lineas[-1].endswith("\n"):
        lineas = lineas[:-1]
    return np.genfromtxt(lineas, delimiter=",", names=True)


BLOQUE_K = 20.0          # K, bloques de la curva E–T
VENTANA_K = 100.0        # K, semiancho de la regresión de C_P
PROMINENCIA = 2.0        # k_B/át: cuánto tiene que sobresalir un pico de C_P (ruido de ±1–2 con 126 átomos)


def guardar(fig, ruta):
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"  figura: {ruta}")


def media_movil(x, ancho):
    """Media móvil centrada de 'ancho' puntos (los extremos con ventana más corta)."""
    ancho = max(1, int(ancho))
    c = np.cumsum(np.insert(x, 0, 0.0))
    i = np.arange(len(x))
    lo, hi = np.clip(i - ancho // 2, 0, len(x)), np.clip(i + ancho // 2 + 1, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)


def cp_local(T, H, centros, w):
    """C_P = pendiente de H frente a T en ventanas [c − w, c + w], con sumas acumuladas (O(n log n))."""
    o = np.argsort(T)
    T, H = T[o], H[o]
    T0, H0 = T.mean(), H.mean()
    t, h = T - T0, H - H0
    c1, ct, ch, ctt, cth = (np.insert(np.cumsum(v), 0, 0.0) for v in (np.ones_like(t), t, h, t * t, t * h))
    lo = np.searchsorted(T, centros - w, side="left")
    hi = np.searchsorted(T, centros + w, side="right")
    n, st, sh, stt, sth = (c[hi] - c[lo] for c in (c1, ct, ch, ctt, cth))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (sth - st * sh / n) / (stt - st * st / n) / units.kB          # k_B por átomo


# ----------------------------------------------------------------------------------------------
def figuras_equilibrado(salida, A_ref, n):
    d = leer_csv(f"{salida}/log.csv")
    if d.size < 3:
        return
    print(f"\nEquilibrado ({salida}): {d['t_ps'][-1]:.2f} de {EQUIL_PS:g} ps")
    t = d["t_ps"]
    w = round(500 / LOG_FS)                                      # 0.5 ps
    sig = T_EQUIL * np.sqrt(2 / (3 * n - 3))
    seg = t >= max(EQUIL_PS / 2, TERMALIZACION_PS)
    fig, ax = plt.subplots(3, 2, figsize=(11, 9))
    a = ax[0, 0]
    a.plot(t, d["T_K"], lw=0.4, color="0.7", label="instantánea")
    a.plot(t, media_movil(d["T_K"], w), lw=1.2, label="media 0.5 ps")
    a.axhline(T_EQUIL, color="C3", lw=1, label="consigna")
    a.axhspan(T_EQUIL - sig, T_EQUIL + sig, color="C3", alpha=0.1, label="±σ canónica")
    a.set_ylabel("T (K)"); a.legend(fontsize=7)
    a = ax[0, 1]
    a.plot(t, d["Epot_eV_at"], lw=0.5); a.set_ylabel("E_pot (eV/át)")
    a = ax[1, 0]
    a.plot(t, 100 * (d["area_A2_at"] * n / A_ref - 1), lw=0.8); a.set_ylabel("Área respecto a 0 K (%)")
    a = ax[1, 1]
    a.plot(t, d["tension_Nm"], lw=0.3, color="0.7")
    a.plot(t, media_movil(d["tension_Nm"], w), lw=1.2)
    a.axhline(TENSION_EXT, color="C3", lw=1); a.set_ylabel("Tensión τ (N/m)")
    a = ax[2, 0]
    if seg.sum() > 10:
        a.hist(d["T_K"][seg], bins=30, density=True, alpha=0.6, label=f"t ≥ {t[seg][0]:.1f} ps")
        x = np.linspace(T_EQUIL - 4 * sig, T_EQUIL + 4 * sig, 200)
        a.plot(x, np.exp(-0.5 * ((x - T_EQUIL) / sig) ** 2) / (sig * np.sqrt(2 * np.pi)), "C3", label="canónica")
        a.legend(fontsize=7)
    a.set_xlabel("T (K)")
    a = ax[2, 1]
    for k, et in [("T_xy_K", "T en el plano"), ("T_z_K", "T fuera del plano"), ("T_B_K", "T del B"), ("T_C_K", "T del C")]:
        a.plot(t, media_movil(d[k], 2 * w), lw=1, label=et)
    a.axhline(T_EQUIL, color="k", lw=0.6, ls=":"); a.set_ylabel("T, media 1 ps (K)"); a.legend(fontsize=7)
    for a in ax[1:, :].ravel()[[0, 1, 3]]:
        a.set_xlabel("t (ps)")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1], ax[2, 1]):
        a.axvline(TERMALIZACION_PS, color="k", lw=0.5, ls=":")
    guardar(fig, f"{salida}/equilibrado.png")
    if os.path.exists(f"{salida}/resumen.json"):
        r = leer_json(f"{salida}/resumen.json")
        print(f"  ⟨T⟩ = {r['T_K']:.0f} ± {r['T_err_K']:.0f} K, σ(T) = {r['sigma_T_K']:.0f} K (canónica "
              f"{r['sigma_T_canonica_K']:.0f}); área {r['area_pct']:+.2f} %, deriva {r['deriva_area_pct_ps']:+.3f} %/ps; "
              f"⟨τ⟩ = {r['tension_Nm']:+.3f} ± {r['tension_err_Nm']:.3f} N/m; avisos: {r['avisos'] or 'ninguno'}")


# ----------------------------------------------------------------------------------------------
def figuras_rampa(salida, salidas_cp, A_ref, n):
    d = leer_csv(f"{salida}/log.csv")
    e = leer_csv(f"{salida}/estructura.csv")
    if d.size < 3:
        return
    dur = (T_FIN - T_INI) / RATE
    print(f"\nRampa ({salida}): t = {d['t_ps'][-1]:.1f} de {dur:g} ps ({100 * d['t_ps'][-1] / dur:.1f} %), "
          f"T objetivo {d['T_obj_K'][-1]:.0f} K")
    if os.path.exists(f"{salida}/progreso.txt"):
        with open(f"{salida}/progreso.txt", encoding="utf-8") as f:
            ult = [l for l in f.read().splitlines() if l.startswith("t ")]
        if ult:
            print("  último progreso: " + ult[-1])
    t, To = d["t_ps"], d["T_obj_K"]
    w = round(1000 / LOG_FS)                                     # 1 ps
    lineas = []

    # --- C_P = dH/dT (todas las semillas juntas) y sus picos ---
    T_all, H_all = [], []
    for s in salidas_cp:
        if os.path.exists(f"{s}/log.csv"):
            ds = leer_csv(f"{s}/log.csv")
            if ds.size > 2:
                T_all.append(ds["T_obj_K"]); H_all.append(ds["H_eV_at"])
    T_all, H_all = np.concatenate(T_all), np.concatenate(H_all)
    T_c = np.arange(T_INI + VENTANA_K, T_all.max() - VENTANA_K + 1e-9, 10.0)
    Cp, picos, prom = np.array([]), np.array([], int), np.array([])
    if len(T_c) >= 3:
        Cp = cp_local(T_all, H_all, T_c, VENTANA_K)
        picos, p = find_peaks(Cp, prominence=PROMINENCIA)
        prom = p["prominences"]
        lineas.append(f"C_P con {len(salidas_cp)} semilla(s): máximo {np.nanmax(Cp):.1f} k_B/át a {T_c[np.nanargmax(Cp)]:.0f} K")
        lineas.append(f"Picos que sobresalen más de {PROMINENCIA:g} k_B/át: "
                      + (", ".join(f"{T_c[k]:.0f} K ({Cp[k]:.1f}, sobresale {pr:.1f})" for k, pr in zip(picos, prom))
                         or "ninguno"))
    else:
        lineas.append(f"C_P: hace falta recorrer al menos {2 * VENTANA_K + 20:g} K de rampa.")
    T_picos = T_c[picos] if len(picos) else np.array([])

    # --- Figura: evolución temporal ---
    fig, ax = plt.subplots(5, 1, figsize=(8, 11), sharex=True)
    ax[0].plot(t, d["Epot_eV_at"], lw=0.4, label="E_pot")
    ax[0].plot(t, d["Etot_eV_at"], lw=0.4, label="E_tot")
    ax[0].set_ylabel("E (eV/át)"); ax[0].legend(fontsize=7)
    ax[1].plot(t, d["T_K"], lw=0.3, color="0.7", label="instantánea")
    ax[1].plot(t, media_movil(d["T_K"], w), lw=1, label="media 1 ps")
    ax[1].plot(t, To, color="C3", lw=1, label="consigna")
    ax[1].set_ylabel("T (K)"); ax[1].legend(fontsize=7)
    ax[2].plot(t, 100 * (d["area_A2_at"] * n / A_ref - 1), lw=0.6)
    ax[2].set_ylabel("Área resp. 0 K (%)")
    ax[3].plot(t, d["tension_Nm"], lw=0.3, color="0.7")
    ax[3].plot(t, media_movil(d["tension_Nm"], w), lw=1)
    ax[3].axhline(TENSION_EXT, color="C3", lw=0.8); ax[3].set_ylabel("τ (N/m)")
    ax[4].plot(t, d["z_rms_A"], lw=0.6, label="z_rms (todos)")
    if e.size > 1:
        ax[4].plot(e["t_ps"], e["h_rms_A"], lw=1, label="h_rms (red mayor)")
    ax[4].set_ylabel("Ondulación (Å)"); ax[4].set_xlabel("t (ps)"); ax[4].legend(fontsize=7)
    guardar(fig, f"{salida}/rampa_tiempo.png")

    # --- Figura: E frente a T y C_P ---
    b = np.floor((To - T_INI) / BLOQUE_K).astype(int)
    b[b < 0] = 0
    cnt = np.bincount(b)
    ok = cnt > 0
    med = lambda x: (np.bincount(b, weights=x) / np.maximum(cnt, 1))[ok]
    T_b = med(d["T_K"])
    fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax[0].plot(d["T_K"], d["Etot_eV_at"], ",", color="0.8")
    ax[0].plot(T_b, med(d["Etot_eV_at"]), "o", ms=2.5, label="E_tot (bloques de 20 K)")
    ax[0].plot(T_b, med(d["Epot_eV_at"]), "o", ms=2.5, label="E_pot (bloques de 20 K)")
    ax[0].set_ylabel("E (eV/át)"); ax[0].legend(fontsize=7)
    if len(Cp):
        ax[1].plot(T_c, Cp, "-")
        ax[1].plot(T_c[picos], Cp[picos], "v", color="C3")
    ax[1].axhline(3, color="0.5", ls=":", label="Dulong–Petit (3 $k_B$)")
    ax[1].set_ylabel("$C_P$ = dH/dT ($k_B$/át)"); ax[1].set_xlabel("T (K)"); ax[1].legend(fontsize=7)
    for a in ax:
        for Tp in T_picos:
            a.axvline(Tp, color="0.4", ls="--", lw=0.7)
    guardar(fig, f"{salida}/rampa_E_T.png")

    # --- Figura: estructura frente a T ---
    if e.size > 1:
        Te = e["T_obj_K"]
        fig, ax = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
        ax[0].plot(Te, e["BC_intactos"], lw=1, label="B–C intactos")
        ax[0].plot(Te, e["coord_ideal"], lw=1, label="coordinación ideal")
        ax[0].set_ylabel("Fracción"); ax[0].legend(fontsize=7)
        ax[1].plot(Te, e["red_mayor"], lw=1, label="en la red mayor")
        ax[1].plot(Te, e["sueltos"], lw=1, label=f"en fragmentos de ≤ {FRAG_PEQ} átomos")
        ax[1].set_ylabel("Fracción de átomos"); ax[1].legend(fontsize=7)
        ax[2].plot(Te, e["h_rms_A"], lw=1); ax[2].set_ylabel("h_rms (Å)")
        ax[3].plot(Te, e["d_min_A"], lw=1)
        ax[3].axhline(D_MIN_PARADA, color="C3", ls=":", lw=0.8)
        ax[3].set_ylabel("d_min (Å)"); ax[3].set_xlabel("T objetivo (K)")
        for a in ax:
            for Tp in T_picos:
                a.axvline(Tp, color="0.4", ls="--", lw=0.7)
        guardar(fig, f"{salida}/rampa_estructura.png")

        # Umbrales y tabla por pico
        def primera(cond):
            k = np.flatnonzero(cond)
            return f"{Te[k[0]]:.0f} K" if len(k) else "no ocurre"
        lineas += [f"Primer B–C roto: {primera(e['BC_intactos'] < 1)} | 10 % rotos: {primera(e['BC_intactos'] < 0.9)} | "
                   f"red partida (< 95 %): {primera(e['red_mayor'] < 0.95)} | > 5 % sueltos: {primera(e['sueltos'] > 0.05)}"]
        if len(T_picos):
            lineas.append("Cambio en cada pico de C_P ([T+50, T+100] − [T−100, T−50] K):")
            lineas.append(f"{'T (K)':>7} {'C_P':>5} {'B–C int.':>9} {'coord.':>8} {'red may.':>9} {'sueltos':>8} "
                          f"{'h_rms Å':>8} {'área %':>7}")
            area_pct = 100 * (d["area_A2_at"] * n / A_ref - 1)
            for k, Tp in zip(picos, T_picos):
                a_, d_ = (Te >= Tp - 100) & (Te <= Tp - 50), (Te >= Tp + 50) & (Te <= Tp + 100)
                la, ld = (To >= Tp - 100) & (To <= Tp - 50), (To >= Tp + 50) & (To <= Tp + 100)
                if a_.any() and d_.any():
                    dd = {c: e[c][d_].mean() - e[c][a_].mean() for c in ("BC_intactos", "coord_ideal", "red_mayor",
                                                                         "sueltos", "h_rms_A")}
                    lineas.append(f"{Tp:7.0f} {Cp[k]:5.1f} {dd['BC_intactos']:+9.3f} {dd['coord_ideal']:+8.3f} "
                                  f"{dd['red_mayor']:+9.3f} {dd['sueltos']:+8.3f} {dd['h_rms_A']:+8.2f} "
                                  f"{area_pct[ld].mean() - area_pct[la].mean():+7.2f}")
            lineas.append("T⁺ de la BGDY: el pico con la caída de B–C intactos y de coordinación. "
                          "Un pico sin cambio estructural es probablemente ruido (más semillas: bash ver_npt.sh figuras --semillas 1 2 3).")
    for l in lineas:
        print("  " + l)
    with open(f"{salida}/analisis.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")


def seguimiento(semillas=None):
    """Figuras y análisis de lo que haya (también con la rampa en marcha).
    semillas: lista de semillas cuyas rampas se juntan para C_P (por defecto, solo SEMILLA)."""
    rel = f"{RESULTADOS}/01_relajacion/relajada.extxyz"
    if not os.path.exists(rel):
        print("Aún no hay relajación (etapa 1)."); return
    ref = read(rel)
    A_ref, n = area(ref), len(ref)
    titulo("SEGUIMIENTO")
    r1 = leer_json(f"{RESULTADOS}/01_relajacion/resumen.json")
    print(f"Relajación: a = {r1['a_A']:.4f} Å, E = {r1['E_eV_at']:.5f} eV/át, fmax = {r1['fmax_eV_A']:.5f} eV/Å, "
          f"K2D = {r1['K2D_Nm']:.1f} N/m")
    if os.path.exists(f"{carpeta_equilibrado()}/log.csv"):
        figuras_equilibrado(carpeta_equilibrado(), A_ref, n)
    if os.path.exists(f"{carpeta_rampa()}/log.csv"):
        figuras_rampa(carpeta_rampa(), [carpeta_rampa(s) for s in (semillas or [SEMILLA])], A_ref, n)



# ==============================================================================================
# Etapa 1. Relajación
# ==============================================================================================
def tension_xx(atoms, calc, pbc):
    a = atoms.copy()
    a.pbc, a.calc = pbc, calc
    return a.get_stress()[0] * a.cell[2, 2] / NM          # N/m


def tension_media(atoms):
    s = atoms.get_stress()
    return 0.5 * (s[0] + s[1]) * atoms.cell[2, 2] / NM    # N/m


def relajacion():
    global PBC_MODELO
    t0 = time.time()
    salida = carpeta(f"{RESULTADOS}/01_relajacion")
    titulo("1. RELAJACIÓN: posiciones + celda en el plano (c fijo), tensión cero, 0 K")

    atoms = read(ESTRUCTURA)
    atoms.pbc = PBC_LAMINA
    calc64 = cargar_modelo("float64")
    atoms.calc = calc64
    n = len(atoms)
    print(f"{atoms.get_chemical_formula()}: {n} átomos | a = {atoms.cell.lengths()[0]:.4f} Å, "
          f"γ = {atoms.cell.angles()[2]:.2f}°, c = {atoms.cell[2, 2]:.1f} Å | modelo {MODELO}, cabecera {HEAD}")

    # --- Comprobación del stress: con mace-torch < 0.3.17 el de z libre está mal → pbc = True ---
    ver, vt = version_mace()
    s_libre = tension_xx(atoms, calc64, PBC_LAMINA)
    s_per = tension_xx(atoms, calc64, True)
    print(f"mace-torch {ver} | stress σxx·c: z libre {s_libre:+.4f} N/m | z periódico {s_per:+.4f} N/m")
    if vt >= (0, 3, 17) and np.isclose(s_libre, s_per, rtol=0.01, atol=1e-4):
        PBC_MODELO = PBC_LAMINA
        print("Stress correcto con z libre: se usa pbc = T T F.")
    else:
        PBC_MODELO = True
        atoms = ampliar_vacio(atoms, VACIO_MIN_PBC)
        atoms.calc = calc64
        print(f"⚠ Con mace-torch {ver} el stress con z libre no es fiable: se usa pbc = T T T en todo el "
              f"cálculo, con c = {atoms.cell[2, 2]:.0f} Å (equivalente: el corte de MACE es 6 Å). Para usar z libre, "
              "actualiza el venv: pip install --no-deps mace-torch==0.3.17")
    atoms.pbc = PBC_MODELO

    # --- BFGS: posiciones + a, b, γ (máscara xx, yy, xy) ---
    e0, a0, A0 = atoms.get_potential_energy() / n, atoms.cell.lengths()[0], area(atoms)
    enl0 = longitudes_enlace(atoms)
    filtro = FrechetCellFilter(atoms, mask=[True, True, False, False, False, True])
    opt = BFGS(filtro, logfile=f"{salida}/bfgs.log", trajectory=f"{salida}/bfgs.traj")
    convergido = opt.run(fmax=FMAX, steps=PASOS_MAX_BFGS)
    fmax = np.linalg.norm(atoms.get_forces(), axis=1).max()
    e1, a1, A1 = atoms.get_potential_energy() / n, atoms.cell.lengths()[0], area(atoms)
    enl1 = longitudes_enlace(atoms)
    tau = tension_media(atoms)
    print(f"BFGS {'convergido' if convergido else 'NO CONVERGIDO'} en {opt.nsteps} pasos: fmax = {fmax:.5f} eV/Å, "
          f"τ = {tau:+.5f} N/m")
    print(f"  a: {a0:.4f} → {a1:.4f} Å ({100 * (a1 / a0 - 1):+.3f} %) | b = {atoms.cell.lengths()[1]:.4f} Å | "
          f"γ = {atoms.cell.angles()[2]:.3f}° | área: {A0:.2f} → {A1:.2f} Å² ({100 * (A1 / A0 - 1):+.3f} %)")
    print(f"  E: {e0:.6f} → {e1:.6f} eV/át ({1000 * (e1 - e0):+.3f} meV/át)")
    print(f"  Enlaces (Å): B–C {enl0['B-C']:.4f} → {enl1['B-C']:.4f} | C≡C {enl0['C≡C']:.4f} → {enl1['C≡C']:.4f} | "
          f"C–C {enl0['C-C']:.4f} → {enl1['C-C']:.4f}  ({enl1['n_BC']} B–C, {enl1['n_CtripleC']} C≡C, {enl1['n_CC']} C–C)")
    print(f"  z: desviación máxima respecto al plano {np.ptp(atoms.positions[:, 2]):.2e} Å")
    if not convergido:
        raise RuntimeError("BFGS no ha convergido: sube PASOS_MAX_BFGS.")
    write(f"{salida}/relajada.extxyz", atoms)

    # --- Módulo 2D con iones relajados: τ(±δ) con celda fija ---
    d = DEFORMACION_K2D
    tensiones = []
    for signo in (+1, -1):
        b = atoms.copy()
        b.calc = calc64
        b.set_cell(atoms.cell.array @ np.diag([1 + signo * d, 1 + signo * d, 1]), scale_atoms=True)
        BFGS(b, logfile=None).run(fmax=5 * FMAX, steps=500)
        tensiones.append(tension_media(b))
    K2D = (tensiones[0] - tensiones[1]) / ((1 + d) ** 2 - (1 - d) ** 2)   # τ = K2D·ΔA/A
    print(f"Módulo 2D (iones relajados, ±{100 * d:g} %): τ = {tensiones[0]:+.3f} / {tensiones[1]:+.3f} N/m "
          f"→ K2D = {K2D:.1f} N/m (masa del pistón con periodo {PERIODO_PISTON:g} ps)")

    # --- Stress del modelo de la dinámica (float32) frente a float64 ---
    s32 = tension_xx(atoms, cargar_modelo(DTYPE_MD, comprobar_en=atoms), PBC_MODELO) \
        if DTYPE_MD != "float64" else tension_xx(atoms, calc64, PBC_MODELO)
    s64 = tension_xx(atoms, calc64, PBC_MODELO)
    print(f"Stress en la lámina relajada: {DTYPE_MD} {s32:+.5f} N/m | float64 {s64:+.5f} N/m")
    if abs(s32 - s64) > 0.01:
        raise RuntimeError(f"El stress en {DTYPE_MD} se separa del de float64: usa DTYPE_MD = 'float64'.")

    guardar_json(f"{salida}/resumen.json", {
        "estructura": ESTRUCTURA, "modelo": MODELO, "head": HEAD, "n_atomos": n,
        "convergido": bool(convergido), "pasos_bfgs": opt.nsteps, "fmax_eV_A": fmax, "tension_Nm": tau,
        "a_ini_A": a0, "a_A": a1, "b_A": atoms.cell.lengths()[1], "gamma_deg": atoms.cell.angles()[2],
        "area_ini_A2": A0, "area_A2": A1, "E_ini_eV_at": e0, "E_eV_at": e1,
        "enlaces_ini_A": enl0, "enlaces_A": enl1, "K2D_Nm": K2D, "tensiones_K2D_Nm": tensiones,
        "stress_xx_zlibre_Nm": s_libre, "stress_xx_zperiodico_Nm": s_per,
        "stress_xx_dtype_md_Nm": s32, "dtype_md": DTYPE_MD, "mace_torch": ver,
        "pbc_modelo": "T T F" if PBC_MODELO is PBC_LAMINA else "T T T", "c_A": atoms.cell[2, 2], "tiempo_s": time.time() - t0})
    print(f"Guardado en {salida}/ ({time.time() - t0:.0f} s).")



# ==============================================================================================
# Etapa 2. Equilibrado
# ==============================================================================================
def diagnostico(salida, ref):
    d = leer_csv(f"{salida}/log.csv")
    e = leer_csv(f"{salida}/estructura.csv")
    m = d["t_ps"] >= max(EQUIL_PS / 2, TERMALIZACION_PS)       # segunda mitad, ya sin fricción alta
    if m.sum() < 20:                                                 # equilibrado muy corto
        m = d["t_ps"] >= d["t_ps"][-1] / 2
    t, n = d["t_ps"][m], len(ref)
    T, sT = media_error(d["T_K"][m])
    sigma_can = T_EQUIL * np.sqrt(2 / (3 * n - 3))
    area_pct = 100 * (d["area_A2_at"][m] * n / area(ref) - 1)
    A, sA = media_error(area_pct)
    deriva = np.polyfit(t, area_pct, 1)[0]
    tau, stau = media_error(d["tension_Nm"][m])
    r = {"ventana_ps": [float(t[0]), float(t[-1])], "T_K": T, "T_err_K": sT, "sigma_T_K": d["T_K"][m].std(),
         "sigma_T_canonica_K": sigma_can, "T_xy_K": d["T_xy_K"][m].mean(), "T_z_K": d["T_z_K"][m].mean(),
         "T_B_K": d["T_B_K"][m].mean(), "T_C_K": d["T_C_K"][m].mean(),
         "Epot_eV_at": d["Epot_eV_at"][m].mean(), "area_pct": A, "area_err_pct": sA, "deriva_area_pct_ps": deriva,
         "tension_Nm": tau, "tension_err_Nm": stau, "z_rms_A": d["z_rms_A"][m].mean(),
         "BC_intactos_final": float(e["BC_intactos"][-1]), "red_mayor_final": float(e["red_mayor"][-1])}
    titulo(f"Diagnóstico del equilibrado ({t[0]:.1f}–{t[-1]:.1f} ps)")
    print(f"⟨T⟩ = {T:.0f} ± {sT:.0f} K (consigna {T_EQUIL:g}) | σ(T) = {r['sigma_T_K']:.0f} K "
          f"(canónica {sigma_can:.0f} K)")
    print(f"Equipartición: T_xy {r['T_xy_K']:.0f} K, T_z {r['T_z_K']:.0f} K | T_B {r['T_B_K']:.0f} K, "
          f"T_C {r['T_C_K']:.0f} K")
    print(f"Área: {A:+.2f} ± {sA:.2f} % respecto a 0 K, deriva {deriva:+.3f} %/ps | "
          f"tensión ⟨τ⟩ = {tau:+.3f} ± {stau:.3f} N/m (objetivo {TENSION_EXT:g})")
    print(f"E_pot = {r['Epot_eV_at']:.5f} eV/át | z_rms = {r['z_rms_A']:.2f} Å | "
          f"B–C intactos {r['BC_intactos_final']:.1%}, red mayor {r['red_mayor_final']:.1%} (último frame)")
    avisos = []
    if abs(T - T_EQUIL) > 3 * sT + 0.01 * T_EQUIL:
        avisos.append("⟨T⟩ se separa de la consigna")
    if abs(deriva) > 0.2:
        avisos.append("el área aún deriva más de 0.2 %/ps: alarga EQUIL_PS")
    if abs(tau - TENSION_EXT) > 3 * stau + 0.05:
        avisos.append("la tensión media no está en la consigna")
    if r["BC_intactos_final"] < 1:
        avisos.append("ya hay enlaces B–C rotos al final del equilibrado")
    print("Avisos: " + "; ".join(avisos) if avisos else "Sin avisos: listo para la rampa.")
    r["avisos"] = avisos
    return r


def equilibrado(reanudar=False, parar=None):
    t_reloj = time.time()
    salida = carpeta_equilibrado()
    titulo(f"2. EQUILIBRADO: {EQUIL_PS:g} ps a {T_EQUIL:g} K, Langevin NPT en el plano (semilla {SEMILLA})")
    integ, seg, t0, ref = preparar_dinamica(salida, reanudar, T_EQUIL, FRICCION_INICIAL)
    estado, t = ejecutar_md(
        integ, seg, t0, EQUIL_PS, consigna=lambda t: T_EQUIL, progreso_ps=PROGRESO_EQUIL_PS,
        friccion=lambda t: FRICCION_INICIAL if t < TERMALIZACION_PS else FRICCION, parar=parar)
    seg.cerrar()
    if estado == "interrumpida":
        return estado
    copiar_final(salida)
    r = diagnostico(salida, ref)
    r.update(estado=estado, t_final_ps=t, semilla=SEMILLA, tiempo_s=time.time() - t_reloj)
    guardar_json(f"{salida}/resumen.json", r)
    print(f"Equilibrado {estado} ({time.time() - t_reloj:.0f} s). Estado final: {salida}/final.extxyz")
    return estado



# ==============================================================================================
# Etapa 3. Rampa
# ==============================================================================================
def rampa(reanudar=False, parar=None, eco=True):
    """Rampa de T_INI a T_FIN a RATE K/ps. parar: threading.Event para detenerla desde fuera; eco=False no
    imprime (todo va a progreso.txt). Devuelve el estado final."""
    T_INI_, T_FIN_, RATE_ = T_INI, T_FIN, RATE                     # fijos aunque se cambien los parámetros
    decir = print if eco else (lambda *a, **k: None)
    t_reloj = time.time()
    duracion = (T_FIN_ - T_INI_) / RATE_
    salida = carpeta_rampa()
    cabecera = (f"3. RAMPA: {T_INI_:g} → {T_FIN_:g} K a {RATE_:g} K/ps ({duracion:g} ps, "
                f"{round(duracion * 1000 / DT_FS):,} pasos), Langevin NPT en el plano (semilla {SEMILLA})")
    if eco:
        titulo(cabecera)
    if T_EQUIL != T_INI_:
        decir(f"⚠ El equilibrado fue a {T_EQUIL:g} K y la rampa empieza a {T_INI_:g} K.")

    def consigna(t):
        return min(T_INI_ + RATE_ * t, T_FIN_)

    integ, seg, t0, ref = preparar_dinamica(salida, reanudar, consigna(0.0), FRICCION,
                                            inicio=f"{carpeta_equilibrado()}/final.extxyz", eco=eco)
    if not eco:
        seg.mensaje(cabecera)
    estado, t = ejecutar_md(integ, seg, t0, duracion, consigna, progreso_ps=PROGRESO_RAMPA_PS, parar=parar)
    seg.cerrar()
    if estado == "interrumpida":
        return estado
    copiar_final(salida)

    # --- Resumen estructural: primera T a la que cada indicador cruza su umbral ---
    e = leer_csv(f"{salida}/estructura.csv")

    def primera_T(cond):
        k = np.flatnonzero(cond)
        return float(e["T_obj_K"][k[0]]) if len(k) else None

    cruces = {"primer B–C roto": primera_T(e["BC_intactos"] < 1),
              "10 % de los B–C rotos": primera_T(e["BC_intactos"] < 0.9),
              "la red se parte (< 95 % en la mayor)": primera_T(e["red_mayor"] < 0.95),
              "más del 5 % de átomos sueltos": primera_T(e["sueltos"] > 0.05)}
    lineas = [f"Rampa {estado}. Primera T a la que cruza cada umbral:"] + \
        [f"  {k:38s}: {'no ocurre' if v is None else f'{v:.0f} K'}" for k, v in cruces.items()]
    with open(f"{salida}/progreso.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")
    decir("\n".join(lineas))
    guardar_json(f"{salida}/resumen.json", {"estado": estado, "t_final_ps": t, "T_final_K": consigna(t),
                                               "cruces_K": cruces, "semilla": SEMILLA,
                                               "tiempo_s": time.time() - t_reloj})
    return estado



# ----------------------------------------------------------------------------------------------
# Ejecución en LUMI: modos, señales y códigos de salida
# ----------------------------------------------------------------------------------------------
PARAR = threading.Event()
CODIGO = {"completa": 0, "interrumpida": 3, "parada por colapso": 4}


def _senal(signum, frame):
    if not PARAR.is_set():
        print(f"\n>>> Señal {signal.Signals(signum).name}: se para al final del bloque en curso y se guarda un frame.",
              flush=True)
    PARAR.set()


def entorno():
    titulo(f"rampa_npt_bgdy.py | {time.strftime('%Y-%m-%d %H:%M:%S')} | {socket.gethostname()} | "
           f"trabajo SLURM {os.environ.get('SLURM_JOB_ID', '-')}")
    try:
        import torch, ase, mace
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO HAY GPU"
        print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | mace-torch {mace.__version__} | "
              f"ASE {ase.__version__} | GPU: {gpu} | hilos: {torch.get_num_threads()}")
        if DEVICE == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DEVICE = 'cuda' pero no hay GPU visible: revisa --gpus-per-task en el trabajo.")
    except ImportError as e:
        print("Falta un paquete:", e)
    falta = [f for f in (ESTRUCTURA, MODELO) if not os.path.exists(f)]
    if falta:
        raise FileNotFoundError(f"Faltan en {os.getcwd()}: {falta}")
    print(f"Carpeta: {os.getcwd()} | resultados en {RESULTADOS}/ | semilla {SEMILLA}")


def todo():
    """Hace lo que falte, reanudando lo que esté a medias. Devuelve el estado final."""
    if os.path.exists(f"{RESULTADOS}/01_relajacion/relajada.extxyz"):
        print(f"Relajación ya hecha: {RESULTADOS}/01_relajacion/relajada.extxyz")
    else:
        relajacion()
    eq = carpeta_equilibrado()
    if os.path.exists(f"{eq}/final.extxyz"):
        print(f"Equilibrado ya hecho: {eq}/final.extxyz")
    else:
        est = equilibrado(reanudar=os.path.exists(f"{eq}/frames.extxyz"), parar=PARAR)
        if est != "completa":
            return est
    rp = carpeta_rampa()
    hecho = leer_json(f"{rp}/resumen.json").get("estado") if os.path.exists(f"{rp}/resumen.json") else None
    if hecho in ("completa", "parada por colapso"):
        print(f"Rampa ya terminada ({hecho}): {rp}")
        est = hecho
    else:
        est = rampa(reanudar=os.path.exists(f"{rp}/frames.extxyz"), parar=PARAR)
        if est == "interrumpida":
            return est
    seguimiento()
    return est


def prueba():
    """Relajación + PRUEBA_EQUIL_PS de equilibrado + PRUEBA_RAMPA_PS de rampa en resultados_prueba/."""
    global RESULTADOS, EQUIL_PS, TERMALIZACION_PS, PROGRESO_EQUIL_PS, T_FIN, PROGRESO_RAMPA_PS
    equil_real, T_fin_real = EQUIL_PS, T_FIN
    RESULTADOS = "resultados_prueba"
    shutil.rmtree(RESULTADOS, ignore_errors=True)
    EQUIL_PS, TERMALIZACION_PS, PROGRESO_EQUIL_PS = PRUEBA_EQUIL_PS, PRUEBA_EQUIL_PS / 2, PRUEBA_EQUIL_PS / 4
    T_FIN, PROGRESO_RAMPA_PS = T_INI + RATE * PRUEBA_RAMPA_PS, PRUEBA_RAMPA_PS / 5
    relajacion()
    est = equilibrado(parar=PARAR)
    if est == "completa":
        est = rampa(parar=PARAR)
    with open(f"{carpeta_rampa()}/progreso.txt", encoding="utf-8") as f:
        ms = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?) ms/paso", f.read())]
    if ms:
        ms = float(np.median(ms))
        pasos_eq = equil_real * 1000 / DT_FS
        pasos_rampa = (T_fin_real - T_INI) / RATE * 1000 / DT_FS
        h_eq, h_rampa = ms * pasos_eq / 3.6e6, ms * pasos_rampa / 3.6e6
        titulo("Estimación para el cálculo completo")
        print(f"{ms:.0f} ms/paso → equilibrado {h_eq:.1f} h + rampa {h_rampa:.1f} h = {h_eq + h_rampa:.1f} h "
              f"({pasos_rampa:,.0f} pasos de rampa)")
        n_trab = int(np.ceil((h_eq + h_rampa) / 47.5))
        print("Cabe en un trabajo de 2 días." if n_trab == 1 else
              f"Harán falta ~{n_trab} trabajos de 2 días: cada uno se para 15 min antes del límite y se reenvía "
              "solo (hasta 6 en total; MAX_REENVIOS en rampa_npt_lumi.sh).")
    return est


def estado_cli(n=6):
    """Últimas líneas de progreso de cada etapa (no carga el modelo ni usa la GPU)."""
    for nombre, c in (("Equilibrado", carpeta_equilibrado()), ("Rampa", carpeta_rampa())):
        prog = f"{c}/progreso.txt"
        print(f"\n--- {nombre}: {c}")
        if not os.path.exists(prog):
            print("(sin empezar)")
            continue
        with open(prog, encoding="utf-8") as f:
            print("\n".join(f.read().splitlines()[-n:]))
        if os.path.exists(f"{c}/resumen.json"):
            print(f"resumen.json: estado = {leer_json(f'{c}/resumen.json').get('estado')}")


def main():
    p = argparse.ArgumentParser(description="Rampa térmica de BGDY con MACE-MH-1 y Langevin NPT (LUMI-G).")
    p.add_argument("modo", nargs="?", default="todo",
                   choices=["todo", "prueba", "relajacion", "equilibrado", "rampa", "figuras", "estado"])
    p.add_argument("--reanudar", action="store_true", help="equilibrado/rampa: seguir desde el último frame")
    p.add_argument("--semillas", type=int, nargs="*", default=None, help="figuras: semillas que se juntan en C_P")
    a = p.parse_args()
    if a.modo == "estado":
        estado_cli(); return 0
    if a.modo == "figuras":
        seguimiento(a.semillas); return 0
    for s in ("SIGUSR1", "SIGTERM"):
        if hasattr(signal, s):
            signal.signal(getattr(signal, s), _senal)
    entorno()
    if a.modo == "todo":
        est = todo()
    elif a.modo == "prueba":
        est = prueba()
    elif a.modo == "relajacion":
        relajacion(); est = "completa"
    elif a.modo == "equilibrado":
        est = equilibrado(reanudar=a.reanudar, parar=PARAR)
    else:
        est = rampa(reanudar=a.reanudar, parar=PARAR)
    print(f"\nFin: {est} (código de salida {CODIGO.get(est, 1)}) | {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    return CODIGO.get(est, 1)


if __name__ == "__main__":
    sys.exit(main())
