#!/bin/bash
#SBATCH --output=out.out
#SBATCH --partition=brown
#SBATCH --time=08:00:00
#SBATCH --gres=gpu



python ddpm_train.py