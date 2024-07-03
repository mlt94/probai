#!/bin/bash
#SBATCH --output=install.out
#SBATCH --partition=brown
#SBATCH --time=08:00:00
#SBATCH --gres=gpu

module load Anaconda3

#conda create -n ddpm python=3.10.10
source activate ddpm

#conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
#pip install matplotlib numpy tqdm tensorboard scipy einops

#pip install notebook ipython
pip install jupyter

