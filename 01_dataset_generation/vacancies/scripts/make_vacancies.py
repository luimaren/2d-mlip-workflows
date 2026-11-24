#!/usr/bin/env python3
"""
make_vacancies.py -- genera el set de vacantes de C (n <= 4) para una supercelda de
boron-graphdiyne (BGDY) perfecta.

Estructura: los B son nodos 3-coordinados de un panal y estan unidos por linkers
diacetilenicos  B - Ca = Cb - Cb' = Ca' - B  (Ca: C enlazado a B; Cb: C central).
El script reconstruye el grafo de enlaces con PBC, identifica linkers y poros
(hexagonos) y construye los casos de vacantes a partir de ellos, de modo que no
depende de indices escritos a mano.  Todos los defectos locales se colocan en el
poro mas cercano al centro de la celda (no cruzan el borde).

Uso:
    python make_vacancies.py bgdy_3x3.xyz -o vacancies/ [--n-random 3] [--seed 1] [--plot]

Salida (en -o):
    case_XX_<nombre>.xyz   un extxyz por caso (orden de atomos original, sin los eliminados)
    all_cases.xyz          todos los casos en un solo fichero multi-frame
    manifest.csv           indices eliminados (1-based, orden del xyz) y diagnostico de cada caso
    overview.png           (con --plot) esquema de las vacantes de cada caso
"""
import argparse
import csv
import itertools
import os
import sys

import numpy as np
from ase.geometry import get_distances
from ase.io import read, write
from ase.neighborlist import neighbor_list
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

BOND_CUT = 1.8   # A. B-C = 1.5, C-C = 1.24/1.34 en la estructura perfecta; no-enlazados > 2.4


# ----------------------------------------------------------------------------- grafo
def build_graph(atoms):
    sym = np.array(atoms.get_chemical_symbols())
    i, j, S = neighbor_list("ijS", atoms, BOND_CUT)
    nbrs = [[] for _ in range(len(atoms))]
    for a, b, s in zip(i, j, S):
        nbrs[a].append((int(b), tuple(int(x) for x in s)))
    for k, n in enumerate(nbrs):
        want = 3 if sym[k] == "B" else 2
        if len(n) != want:
            sys.exit("ERROR: el atomo %d (%s) tiene %d vecinos (esperaba %d). "
                     "Esto no parece la celda BGDY perfecta." % (k + 1, sym[k], len(n), want))
    return sym, nbrs


def build_linkers(sym, nbrs):
    """Cada linker: atomos [a_s, b_s, b_e, a_e] (de Bi a Bj), Bi, Bj y desplazamiento S de Bj."""
    linkers, seen = [], {}
    for b in np.where(sym == "B")[0]:
        for a, s0 in nbrs[b]:
            chain, prev, cur, hop = [a], b, a, np.array(s0)
            off = np.array(s0)
            while True:
                cand = [(k, s) for k, s in nbrs[cur]
                        if not (k == prev and np.array_equal(s, -hop))]
                assert len(cand) == 1
                k, s = cand[0]
                off = off + np.array(s)
                if sym[k] == "B":
                    end = k
                    break
                chain.append(k)
                prev, cur, hop = cur, k, np.array(s)
            if len(chain) != 4:
                sys.exit("ERROR: linker de %d C (esperaba 4)." % len(chain))
            key = frozenset(chain)
            if key not in seen:
                seen[key] = len(linkers)
                linkers.append(dict(atoms=chain, Bi=int(b), Bj=int(end), S=off))
    return linkers


def build_adjacency(linkers):
    """adj[B] = lista de entradas (Bj, S, lid, dir); dir=+1 si el linker se recorre en su sentido guardado."""
    adj = {}
    for lid, L in enumerate(linkers):
        adj.setdefault(L["Bi"], []).append((L["Bj"], tuple(L["S"]), lid, +1))
        adj.setdefault(L["Bj"], []).append((L["Bi"], tuple(-L["S"]), lid, -1))
    return adj


def chain_of(linkers, entry):
    """Atomos del linker orientados en el sentido del recorrido de la entrada."""
    _, _, lid, d = entry
    c = linkers[lid]["atoms"]
    return list(c) if d > 0 else list(c[::-1])


def find_pores(atoms, adj, linkers):
    """Hexagonos contractiles (suma de desplazamientos = 0). Cada poro: B, entradas ordenadas y centro."""
    pos, cell = atoms.get_positions(), atoms.cell.array
    pores, found = [], set()

    def dfs(start, path_B, path_E, off):
        cur = path_B[-1]
        if len(path_E) == 6:
            if cur == start and not off.any():
                key = frozenset(e[2] for e in path_E)
                if key not in found:
                    found.add(key)
                    # centro: media de las posiciones sin envolver de los 6 B
                    o, pts = np.zeros(3), []
                    for b, e in zip(path_B[:6], path_E):
                        pts.append(pos[b] + o @ cell)
                        o = o + np.array(e[1])
                    pores.append(dict(B=list(path_B[:6]), E=list(path_E), center=np.mean(pts, axis=0)))
            return
        for e in adj[cur]:
            nb = e[0]
            if nb in path_B[1:] and not (len(path_E) == 5 and nb == start):
                continue
            if nb == start and len(path_E) != 5:
                continue
            dfs(start, path_B + [nb], path_E + [e], off + np.array(e[1]))

    for b in adj:
        dfs(b, [b], [], np.zeros(3, dtype=int))
    return pores


# ----------------------------------------------------------------------------- utilidades
def mic_dist(atoms, p, q):
    return get_distances(np.atleast_2d(p), np.atleast_2d(q), cell=atoms.cell, pbc=atoms.pbc)[1][0, 0]


def analyze(sym, nbrs, removed, atoms):
    """Diagnostico del defecto: B subcoordinados, C colgantes, fragmentos sueltos y separacion minima."""
    n = len(sym)
    keep = np.ones(n, bool)
    keep[list(removed)] = False
    rows, cols = [], []
    deg = np.zeros(n, int)
    for a in range(n):
        if not keep[a]:
            continue
        for b, _ in nbrs[a]:
            if keep[b]:
                rows.append(a)
                cols.append(b)
                deg[a] += 1
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    ncomp, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab[keep])
    fragments = int((sizes > 0).sum()) - 1
    kinds = ["a" if any(sym[k] == "B" for k, _ in nbrs[r]) else "b" for r in removed]
    pos = atoms.get_positions()
    mind = min((mic_dist(atoms, pos[i], pos[j]) for i, j in itertools.combinations(removed, 2)), default=float("nan"))
    idx = np.where(keep & (sym == "B"))[0]
    return dict(
        n=len(removed), n_alpha=kinds.count("a"), n_beta=kinds.count("b"),
        B_under=int(sum(deg[b] < 3 for b in idx)), B_min=int(min(deg[b] for b in idx)),
        dangling_C=int(((deg == 1) & keep & (sym == "C")).sum()),
        fragments=fragments, min_dist=mind,
    )


def valid_random(info):
    return info["fragments"] == 0 and info["B_min"] >= 2


# ----------------------------------------------------------------------------- casos
def build_cases(atoms, sym, nbrs, linkers, adj, pores, n_random, seed):
    pos, cell = atoms.get_positions(), atoms.cell.array
    inv = np.linalg.inv(cell)

    # poro mas cercano al centro de la celda
    def dist_to_center(p):
        f = (p["center"] @ inv)
        f[:2] -= np.floor(f[:2])
        return np.linalg.norm((f - np.array([0.5, 0.5, f[2]])) @ cell)
    P = min(pores, key=dist_to_center)
    L = [chain_of(linkers, e) for e in P["E"]]       # L[k]: linker k orientado B_k -> B_{k+1}
    Bs, E = P["B"], P["E"]                           # [a_s, b_s, b_e, a_e] = L[k][0..3]

    betas = [a for a in range(len(sym)) if sym[a] == "C" and not any(sym[k] == "B" for k, _ in nbrs[a])]

    def closest_pair(A, B):
        return min(itertools.product(A, B), key=lambda t: mic_dist(atoms, pos[t[0]], pos[t[1]]))

    def midpoint(lid):
        return pos[linkers[lid]["atoms"]].mean(axis=0)

    cases = []
    add = lambda name, idx, desc: cases.append((name, [int(i) for i in idx], desc))

    # n = 1
    add("V1_alpha", [L[0][0]], "C enlazado a B: B subcoordinado y muñon de 3 C")
    add("V1_beta", [L[0][1]], "C central: linker cortado por la mitad")

    # n = 2
    add("V2_beta-beta_same-linker", [L[0][1], L[0][2]], "centro del linker (enlace simple): dos muñones de 1 C")
    add("V2_alpha-beta_same-end", [L[0][0], L[0][1]], "triple enlace Ca=Cb: muñon de 2 C")
    add("V2_alpha-alpha_same-B", [L[0][3], L[1][0]], "dos Ca de linkers que comparten B: B con un solo vecino")
    add("V2_beta-beta_adjacent", [L[0][2], L[1][1]], "Cb de linkers adyacentes (comparten B), mismo poro")
    pa, pb = closest_pair([L[0][1], L[0][2]], [L[3][1], L[3][2]])
    add("V2_beta-beta_para", [pa, pb], "Cb de linkers opuestos del mismo poro")
    far = max(itertools.combinations(betas, 2), key=lambda t: mic_dist(atoms, pos[t[0]], pos[t[1]]))
    add("V2_beta-beta_far", far, "Cb + Cb lo mas lejos posible (referencia sin interaccion)")

    # n = 3
    add("V3_linker-3C", [L[0][0], L[0][1], L[0][2]], "3 C de un linker: queda un solo C pegado a un B")
    add("V3_pore-alternate", [L[k][1] for k in (0, 2, 4)], "Cb en 3 linkers alternos de un poro (simetria C3)")
    add("V3_pore-consecutive", [L[k][1] for k in range(3)], "Cb en 3 linkers consecutivos del mismo poro")
    # (no se incluye cortar los 3 linkers de un B: deja una molecula BC3 suelta)

    # n = 4
    add("V4_missing-linker", L[0], "linker completo eliminado: fusiona dos poros")
    add("V4_pore-consecutive", [L[k][1] for k in range(4)], "Cb en 4 linkers consecutivos del mismo poro")
    # zigzag de 4 linkers (giros alternos izq/der) -> defecto lineal
    ev = lambda b, e: pos[e[0]] + np.array(e[1]) @ cell - pos[b]
    path, cur, e0 = [E[0]], Bs[1], E[0]
    prev_v = ev(Bs[0], E[0])
    for t in range(1, 4):
        want = +1 if t % 2 else -1
        cand = [e for e in adj[cur] if e[2] != path[-1][2]]
        sel = [e for e in cand if np.sign(prev_v[0] * ev(cur, e)[1] - prev_v[1] * ev(cur, e)[0]) == want]
        e = sel[0]
        path.append(e)
        prev_v, cur = ev(cur, e), e[0]
    add("V4_zigzag-line", [chain_of(linkers, e)[1] for e in path], "4 Cb en linkers consecutivos en zigzag (defecto lineal)")
    mids = {lid: midpoint(lid) for lid in range(len(linkers))}
    l1, l2 = max(itertools.combinations(range(len(linkers)), 2),
                 key=lambda t: mic_dist(atoms, mids[t[0]], mids[t[1]]))
    add("V4_two-cuts_far", linkers[l1]["atoms"][1:3] + linkers[l2]["atoms"][1:3],
        "dos cortes Cb+Cb' lejanos (aditividad / localidad)")

    # n = 4, aleatorias (test set)
    Cs = np.where(sym == "C")[0]
    rng = np.random.default_rng(seed)
    done = {frozenset(c[1]) for c in cases}
    tries = 0
    while sum(c[0].startswith("V4_rand") for c in cases) < n_random:
        tries += 1
        if tries > 100000:
            sys.exit("ERROR: no consigo generar configuraciones aleatorias validas.")
        pick = [int(x) for x in rng.choice(Cs, 4, replace=False)]
        if frozenset(pick) in done:
            continue
        info = analyze(sym, nbrs, pick, atoms)
        if not valid_random(info) or info["n_alpha"] == 4 or info["n_beta"] == 4:
            continue                       # exige mezcla alpha/beta, sin B aislado ni fragmentos
        if any(set(pick) == set(L_["atoms"]) for L_ in linkers):
            continue
        done.add(frozenset(pick))
        k = sum(c[0].startswith("V4_rand") for c in cases) + 1
        add("V4_random-%d" % k, sorted(pick), "4 C aleatorios (mezcla alpha/beta), semilla %d: reservar como test" % seed)
    return cases


# ----------------------------------------------------------------------------- dibujo
def make_plot(atoms, sym, nbrs, cases, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pos, cell = atoms.get_positions(), atoms.cell.array
    ncol = 4
    nrow = int(np.ceil(len(cases) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.9 * nrow))
    axs = np.atleast_1d(axs).ravel()
    corners = np.array([[0, 0], cell[0, :2], cell[0, :2] + cell[1, :2], cell[1, :2], [0, 0]])
    for ax, (name, idx, _) in zip(axs, cases):
        for a in range(len(atoms)):                     # medios enlaces (respeta PBC)
            for b, s in nbrs[a]:
                d = pos[b] + np.array(s) @ cell - pos[a]
                dead = a in idx or b in idx
                ax.plot([pos[a, 0], pos[a, 0] + d[0] / 2], [pos[a, 1], pos[a, 1] + d[1] / 2],
                        color="#e8a0a0" if dead else "0.65", lw=0.6, ls=":" if dead else "-", zorder=1)
        c = np.array([sym[k] == "B" for k in range(len(sym))])
        ax.scatter(*pos[~c, :2].T, s=7, color="0.45", zorder=2)
        ax.scatter(*pos[c, :2].T, s=26, color="#f4a6a6", edgecolor="0.3", lw=0.4, zorder=3)
        ax.scatter(*pos[idx, :2].T, s=70, facecolor="none", edgecolor="red", lw=1.6, zorder=4)
        ax.plot(*corners.T, color="k", lw=0.7)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(name.replace("_", " ", 1), fontsize=9)
    for ax in axs[len(cases):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xyz", help="supercelda BGDY perfecta (xyz/extxyz con Lattice)")
    ap.add_argument("-o", "--outdir", default="vacancies")
    ap.add_argument("--n-random", type=int, default=3, help="configuraciones aleatorias de 4 vacantes (def. 3)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--plot", action="store_true", help="genera overview.png")
    a = ap.parse_args()

    atoms = read(a.xyz)
    if not atoms.cell.volume:
        sys.exit("ERROR: el xyz no trae celda (falta Lattice=...).")
    sym, nbrs = build_graph(atoms)
    linkers = build_linkers(sym, nbrs)
    adj = build_adjacency(linkers)
    pores = find_pores(atoms, adj, linkers)
    nB, nC = (sym == "B").sum(), (sym == "C").sum()
    print("Celda: %d B + %d C | %d linkers | %d poros" % (nB, nC, len(linkers), len(pores)))
    if len(linkers) != 3 * nB // 2 or len(pores) != nB // 2:
        sys.exit("ERROR: topologia inesperada (se esperan %d linkers y %d poros)." % (3 * nB // 2, nB // 2))

    cases = build_cases(atoms, sym, nbrs, linkers, adj, pores, a.n_random, a.seed)
    os.makedirs(a.outdir, exist_ok=True)

    rows, frames = [], []
    pristine = atoms.copy()
    pristine.info["case"] = "pristine"
    write(os.path.join(a.outdir, "case_00_pristine.xyz"), pristine)
    frames.append(pristine)
    hdr = "%-3s %-26s %2s %2s %2s  %-12s %-9s %-8s %-9s %s" % (
        "#", "caso", "n", "Ca", "Cb", "B<3(min)", "C-colg.", "fragm.", "d_min(A)", "atomos eliminados (1-based)")
    print("\n" + hdr)
    for k, (name, idx, desc) in enumerate(cases, 1):
        info = analyze(sym, nbrs, idx, atoms)
        keep = np.ones(len(atoms), bool)
        keep[idx] = False
        new = atoms[keep]
        new.info["case"] = name
        write(os.path.join(a.outdir, "case_%02d_%s.xyz" % (k, name)), new)
        frames.append(new)
        lst = " ".join(str(i + 1) for i in sorted(idx))
        if info["fragments"]:
            print("   AVISO: el caso %s deja %d fragmento(s) suelto(s) (revisalo)" % (name, info["fragments"]))
        print("%-3d %-26s %2d %2d %2d  %-12s %-9d %-8d %-9s %s" % (
            k, name, info["n"], info["n_alpha"], info["n_beta"],
            "%d (%d)" % (info["B_under"], info["B_min"]), info["dangling_C"], info["fragments"],
            "-" if np.isnan(info["min_dist"]) else "%.2f" % info["min_dist"], lst))
        rows.append(dict(case=k, name=name, description=desc, n_vac=info["n"], n_alpha=info["n_alpha"],
                         n_beta=info["n_beta"], B_undercoord=info["B_under"], B_min_coord=info["B_min"],
                         dangling_C=info["dangling_C"], fragments=info["fragments"],
                         min_vac_dist_A=round(float(info["min_dist"]), 3) if not np.isnan(info["min_dist"]) else "",
                         removed_atoms_1based=lst, n_atoms_left=int(keep.sum())))
    with open(os.path.join(a.outdir, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write(os.path.join(a.outdir, "all_cases.xyz"), frames)
    if a.plot:
        make_plot(atoms, sym, nbrs, cases, os.path.join(a.outdir, "overview.png"))
    print("\n%d casos escritos en %s/ (manifest.csv, all_cases.xyz%s)" % (
        len(cases), a.outdir, ", overview.png" if a.plot else ""))


if __name__ == "__main__":
    main()
