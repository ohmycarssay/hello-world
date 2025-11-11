#!/bin/bash

#SBATCH --job-name df-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=16G
#SBATCH --time=1-00:00:00
#SBATCH --partition=batch_ugrad
#SBATCH -o logs/slurm-%A-%x.out


echo "job set"

mkdir -p logs
mkdir -p /local_datasets/deepfake_datasets
echo "env set"

source /data/happy5100/anaconda3/etc/profile.d/conda.sh
conda activate happy5100


echo "Task Start"
python -u main.py
echo "Task End"
