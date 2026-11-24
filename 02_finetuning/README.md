# 02 · Fine-tuning of universal MLIPs

Configurations used to fine-tune foundation MLIPs on the BGDY dataset produced in `01_dataset_generation`.

| Subfolder | Content |
|---|---|
| `mace` | Fine-tuning of MACE-MH-1 (`oc20_usemppbe` head) with all weights updated (`config-FTv30-Naive.yml`) or with low-rank adapters (`config-FTv30-LoRA.yml`). Isolated-atom energies (`E0s`) are given explicitly. `FT_F_datasetsize.ipynb` fine-tunes the model with different fractions of the training set and different force weights in the loss, to see how the errors depend on both |
| `mace/committee` | Training of a committee of models that differ only in the random seed, used to estimate the uncertainty of the predictions (see `04_analysis/uncertainty_committee`). `run_finetune.py` launches the training from a YAML file and `submit_finetune.sh` runs it on LUMI-G |
| `nequip` | Fine-tuning configuration for NequIP |
| `pet` | Fine-tuning (full and LoRA) and from-scratch training options for PET with metatrain |
| `tools` | `add_test_total.py` duplicates the test set with `config_type=TOTAL`, so that MACE reports the error on the whole test set in addition to the error per configuration type |

## Example

```bash
mace_run_train --config config-FTv30-Naive.yml
```

The YAML files expect `train.xyz` and `test.xyz` with the keys `energy_qe` and `forces_qe`, and the foundation model `mace-mh-1.model` in the same folder.
