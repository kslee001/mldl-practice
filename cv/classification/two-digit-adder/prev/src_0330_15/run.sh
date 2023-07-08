#!/bin/bash
#SBATCH --job-name=p3
#SBATCH --nodes=1
#SBATCH --exclude=c06
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB

source /home/${USER}/.bashrc
source /home/${USER}/anaconda3/bin/activate
conda activate tch

srun python /home/gyuseonglee/workspace/2day/src/main.py $@
