"""
Lanza el fine-tuning de un modelo MACE.

Debe ejecutarse DENTRO del contenedor ROCm de LUMI con el venv activado,
en un nodo con GPU (via 'srun --pty' para pruebas, o dentro de un sbatch).

Uso:
    python run_finetune.py [ruta_al_config.yml]

Si no se pasa ruta, usa por defecto config-FTv31-Naive.yml
"""

import warnings

warnings.filterwarnings("ignore")

import sys
import logging
from pathlib import Path

import torch

DEFAULT_CONFIG = "config-FTv31-Naive.yml"

# --- Diagnostico de GPU (adaptado a ROCm, sin usar torch.version.cuda) ---

print("GPU disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Numero de GPUs:", torch.cuda.device_count())
    print("Nombre de la GPU:", torch.cuda.get_device_name(0))
print("PyTorch version:", torch.__version__)
print("Version ROCm/HIP:", torch.version.hip)
print("Version MIOpen (via API cudnn):", torch.backends.cudnn.version())


from mace.cli.run_train import main as mace_run_train_main


def train_mace(config_file_path: str) -> None:
    logging.getLogger().handlers.clear()
    sys.argv = ["program", "--config", config_file_path]
    mace_run_train_main()


if __name__ == "__main__":
    if len(sys.argv) > 2:
        sys.exit("Uso: python run_finetune.py [ruta_al_config.yml]")
    config_file = sys.argv[1] if len(sys.argv) == 2 else DEFAULT_CONFIG
    if not Path(config_file).is_file():
        sys.exit(f"No se encuentra el config: {config_file}")
    print(f"Usando config: {config_file}")
    train_mace(config_file)
