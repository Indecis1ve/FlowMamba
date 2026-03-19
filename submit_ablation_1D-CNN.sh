#!/bin/bash
#SBATCH -J 1dcnn_mamba
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH -o logs/1dcnn_%j.out

mkdir -p logs
mkdir -p output/ablation_1dcnn_run1

source /share/home/u2515063027/miniconda3/etc/profile.d/conda.sh
conda activate netmamba
cd /share/home/u2515063027/NetMamba/NetMamba-main

python src/fine-tune_ablation_1D-CNN.py \
    --model net_mamba_classifier \
    --batch_size 128 \
    --epochs 400 \
    --output_dir output/ablation_1dcnn_run1 \
    --log_dir output/ablation_1dcnn_run1 \
    --num_workers 8 \
    --pin_mem
