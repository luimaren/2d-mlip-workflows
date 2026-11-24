# 04 · Analysis

Notebooks and scripts used to analyse MLIP and AIMD trajectories of BGDY.

| Subfolder | What it computes |
|---|---|
| `vdos` | Vibrational density of states from the power spectrum of atomic velocities, with partial contributions (per element, in-plane and out-of-plane) and the participation ratio (`vdos_pr_simple.ipynb`); comparison of 3×3 and 15×15 cells and of short versus long trajectories (`vdos_celdas.py`) |
| `lindemann` | Lindemann parameter in NVT with three definitions (pairwise δ, static and dynamic γ), with the neighbour cutoff chosen from the pair distribution function |
| `bond_breaking` | Thermal stability of the sheet: detection of B–C bond ruptures, classification into dangling-bond defects and topological rearrangements, and rates per bond for each type |
| `ring_statistics` | Ring statistics of the BGDY network as a function of temperature, following the analysis of Nguyen, *Carbon Letters* 29, 521 (2019) |
| `uncertainty_committee` | Uncertainty of a committee of five fine-tuned MACE models (spread of energies and forces), calibrated on part of the data and compared with the true error against DFT, including along an AIMD trajectory at 2250 K |
| `local_structure` | Pair distribution functions, bond angles and rings of a single frame or averaged over a trajectory, for 2D sheets or 3D systems |

Notebooks are distributed without outputs; paths to the trajectories are set in the first cells.
