#!/bin/bash

#SBATCH --job-name=clinicalAD
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH --gres=gpu:A40:1
#SBATCH --output=./log/generate_outputs_%A_%a.log
#SBATCH --error=./log/generate_errors_%A_%a.log
#SBATCH --time=60:00:00
#SBATCH -p qTRDGPU
#SBATCH -A trends53c17

echo "This is a job running on node $(hostname)"
echo "Error output" >&2

conda init
conda activate catalyst
python babywire_train_30channels-1m.py
