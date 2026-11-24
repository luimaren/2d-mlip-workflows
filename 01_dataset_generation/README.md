# 01 · Dataset generation

Workflows that produce the DFT reference data used to fine-tune MLIPs for boron-graphdiyne (BGDY). Each subfolder generates one family of configurations and ends with an extxyz file with energies (`energy_qe`, eV), forces (`forces_qe`, eV/Å), stresses when available (`stress_qe`, eV/Å³, ASE sign convention) and a `config_type` label.

All calculations use Quantum ESPRESSO (`pw.x` 7.5) with PBE + Grimme-D3, PAW pseudopotentials from pslibrary 1.0.0, spin polarisation, a 60 Ry wavefunction cutoff, Gaussian smearing and Γ-point sampling, on hexagonal cells with 20 Å of vacuum. The exact settings of each family are in its input template or notebook.

| Subfolder | Configurations | Entry point |
|---|---|---|
| `aimd_qe` | Ab initio MD (Nosé–Hoover NVT) of the BGDY 3×3 sheet from 500 to 3000 K. Scripts to set up the runs, merge restarted outputs (`output.1`, `output.2`, …) removing duplicated frames, and convert `pw.x` outputs to extxyz | `crear_simulaciones.ipynb`, `merge_md_restarts.py`, `combine_out_qe.py` |
| `aimd_fps_selection` | Selection of uncorrelated AIMD frames: equilibration removed, SOAP descriptors averaged per configuration, PCA, and farthest-point sampling with a quota per temperature. The notes summarise the decisions and figures of the selection | `BGDY_3x3_dataset_AIMD_v4.ipynb`, `BGDY_3x3_AIMD_dataset.md` |
| `strain` | Uniaxial (x, y), biaxial and anisotropic in-plane strain; SCF on each cell | `01_make_strain_v1.ipynb` → `02_submit_scf.sh` → `03_status.sh` → `04_collect.sh` |
| `rattle` | Random atomic displacements of the 3×3 cell at several amplitudes | `make_rattled.ipynb`, `job_scf.sh`, `status.sh`, `collect.py` |
| `rattle_gamma_multicell` | 1000 rattled configurations of ten sheets from 1×1 to 5×5 (14 to 350 atoms), four amplitudes from 0.01 to 0.07 Å, submitted as Slurm job arrays | `genera.py` → `enviar.sh` → `recoge.py` (settings in `config.sh`) |
| `vacancies` | C vacancies (up to four) built from the bond graph of the sheet (linkers and pores identified automatically), relaxation and NVT for each case, with status tracking and restarts | `scripts/make_vacancies.py`, then `01_prepare.sh` … `06_restart.sh` (settings in `workflow.cfg`) |
| `post_rupture_scf` | Configurations sampled after B–C bond rupture in MLIP simulations, selected by farthest-point sampling and recomputed with DFT. Jobs run several cases in sequence, check the remaining time, retry unconverged SCF with a different density mixing and collect only converged cases | `seleccion_fps_postrotura.ipynb`, `01_preparar_casos.ipynb` → `02_enviar_scf.sh` → `03_estado.sh` → `04_recoger.sh` |
| `assembly` | Conversion of every family to a common extxyz format, combined dataset, random train/test split, PCA, t-SNE and UMAP maps, and review of outliers (anomalous interaction energies, unphysical distances, large forces) before exporting the filtered dataset | `convertir_extxyz_dataset.ipynb`, `revision_outliers_BGDY3x3.ipynb` |

## Typical use of an HPC workflow

```bash
./02_submit_scf.sh --dry-run      # show the sbatch commands
./02_submit_scf.sh                # submit pending cases only
./03_status.sh                    # summary of the state of every case
./04_collect.sh                   # converged cases → extxyz
```

The status scripts combine `squeue` with the content of the outputs, so cases that are queued, running or converged are never resubmitted. Pseudopotentials are not included: place the UPF files in `pseudos/` (or edit `pseudo_dir`).
