#!/bin/bash
#SBATCH --output=out.out
#SBATCH --partition=brown
#SBATCH --time=08:00:00
#SBATCH --gres=gpu

module load Anaconda3
source activate ddpm

python /home/mlut/probai/diffusion/ddpm_train.py --cfg