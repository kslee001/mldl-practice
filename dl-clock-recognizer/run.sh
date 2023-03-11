#!/bin/bash
#SBATCH --job-name=prac
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=64000MB

source /home/${USER}/.bashrc
conda activate tch

srun python /home/gyuseonglee/workspace/Incremental/pretrain.py