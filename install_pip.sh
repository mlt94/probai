#!/bin/bash
#SBATCH --output=install.out
#SBATCH --partition=brown
#SBATCH --time=08:00:00

# Create a new Python virtual environment with Python 3.10
module load Python/3.10.4-GCCcore-11.3.0
python3.10 -m venv probai

source probai/bin/activate

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip3 install matplotlib numpy tqdm tensorboard






