# 03 · Molecular dynamics with fine-tuned MLIPs

## `nvt`

Relaxation followed by NVT MD (ASE Nosé–Hoover chain) with a MACE model.

- `md_nvt.py`: relaxation, Maxwell–Boltzmann initial velocities and NVT run. Parameters (model, structure, temperature, time step, number of steps, output intervals) are at the top of the file.
- `md_nvt_continuar.py`: continues a run from the last frame of the `.traj` file with the same integrator, step counter and thermostat variables, which are stored in every frame so that each restart is exact. It stops cleanly before the time limit of the job.
- `submit_md.sh`, `submit_continuar.sh`: Slurm scripts for LUMI-G (ROCm container + virtual environment).
- `conv_traj_xyz.py`: converts the trajectory to extxyz.

## `npt_ramp`

Temperature ramp of the BGDY sheet with in-plane Langevin NPT (z free), used to locate the onset of structural failure.

- `rampa_npt_bgdy.py`: relaxation → equilibration → ramp (e.g. 1500 to 3500 K at 2 K/ps) → figures and analysis (temperature, energy, area, stress, intact B–C bonds, minimum distance, heat capacity). It resumes unfinished stages, supports several replicas (`SEMILLA`), and stops with a specific exit code if the MLIP collapses (two atoms closer than 0.8 Å).
- `rampa_npt_lumi.sh`: Slurm job; when the time limit approaches, Slurm sends `SIGUSR1`, the script saves a frame and exits with code 3, and the job resubmits itself.
- `ver_npt.sh`: progress and figures from the login node, without GPU.

```bash
sbatch --partition=dev-g --time=00:30:00 rampa_npt_lumi.sh prueba   # short test: ms/step and estimated cost
sbatch rampa_npt_lumi.sh                                            # full run
sbatch --export=ALL,SEMILLA=2 rampa_npt_lumi.sh                     # another replica
```

Exit codes: 0 completed, 3 interrupted (resubmitted), 4 stopped by MLIP collapse, 1 error.
