# 2d-mlip-workflows

Workflows and scripts to **build DFT training data, fine-tune universal machine-learning interatomic potentials (MLIPs) and run molecular dynamics (MD) of 2D carbon materials**. The first application is boron-graphdiyne (BGDY, B<sub>18</sub>C<sub>108</sub> 3×3 supercell), a 2D carbon platform for single-atom catalysts and hydrogen storage.

The repository covers the whole chain, from the first-principles reference data to MLIP-driven simulations at length and time scales that DFT cannot reach:

```
DFT / AIMD (Quantum ESPRESSO)  →  configuration selection (SOAP + FPS)  →  curated extxyz dataset
        →  fine-tuning of universal MLIPs (MACE, NequIP, PET)  →  committee uncertainty
        →  large-scale MD (NVT, NPT temperature ramps)  →  structural and vibrational analysis
```

## Contents

| Folder | What it does | Main tools |
|---|---|---|
| [`01_dataset_generation`](01_dataset_generation) | Reference data for BGDY: ab initio MD from 500 to 3000 K with restart merging; SOAP + farthest-point sampling per temperature; strained cells (uniaxial, biaxial, anisotropic); rattled configurations (3×3 cell and 1×1 to 5×5 multicell set at Γ); C vacancies (relaxation → NVT); single-point calculations of post-rupture configurations; final assembly and outlier review | Quantum ESPRESSO 7.5, ASE, dscribe, Slurm |
| [`02_finetuning`](02_finetuning) | Fine-tuning of the MACE-MH-1 foundation model (naive and LoRA), committees for uncertainty quantification, learning curves versus dataset size, NequIP and PET configurations | mace-torch, NequIP, metatrain |
| [`03_molecular_dynamics`](03_molecular_dynamics) | NVT (Nosé–Hoover chain) with exact restarts; in-plane Langevin NPT temperature ramps with automatic resubmission, replicas and detection of MLIP collapse | ASE, mace-torch, Slurm (LUMI-G) |
| [`04_analysis`](04_analysis) | Vibrational density of states, Lindemann parameter, B–C bond-breaking rate, ring statistics, committee uncertainty and local structure | NumPy, SciPy, matscipy, scikit-learn |
| [`structures`](structures) | Relaxed BGDY 3×3 sheet used as starting point | — |

The HPC workflows share the same design: one folder per calculation, Slurm submission scripts that skip finished or queued cases, a status command that classifies each case (pending, running, converged, not converged, interrupted, error) and a collect step that writes an extxyz file ready for training (`energy_qe`, `forces_qe`, `stress_qe`, `config_type`).

## Requirements

- Python ≥ 3.10 with the packages in [`environment.yml`](environment.yml).
- Quantum ESPRESSO 7.5 (`pw.x`) and the PAW pseudopotentials `B.pbe-n-kjpaw_psl.1.0.0.UPF` and `C.pbe-n-kjpaw_psl.1.0.0.UPF` from [pslibrary](https://dalcorso.github.io/pslibrary/) (not distributed here).
- The MACE-MH-1 foundation model (`mace-mh-1.model`), available from the [MACE foundation models](https://github.com/ACEsuit/mace-foundations).

## Data

The curated DFT dataset of BGDY produced with these workflows is openly available on Zenodo: https://doi.org/10.5281/zenodo.24516295.

## Notes

- This is research code, developed for the study of hydrogen adsorption and storage on transition-metal-doped carbon platforms (manuscript in preparation). Code comments and some notebooks are written in Spanish.
- Notebooks are distributed without outputs.

## How to cite

If you use these workflows, please cite the Zenodo record of this repository (see [`CITATION.cff`](CITATION.cff) or the "Cite this repository" button on GitHub).

## Acknowledgements

Developed in the Nanostructures Physics Group of the Universidad de Valladolid within the project "Green hydrogen production on single-atom catalysts and storage on nanohydride platforms" (Agencia Estatal de Investigación, FEDER / European Union). Calculations were performed with allocations on MareNostrum 5 (RES and EuroHPC), LUMI, Leonardo and MeluXina.

## License

[MIT](LICENSE) © 2026 Luis Martín Encinar
