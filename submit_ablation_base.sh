#!/bin/bash
#SBATCH -J base_mamba           ### 作业名称：纯 Mamba 基准
#SBATCH -p gpu                  ### 队列：GPU 队列
#SBATCH -N 1                    ### 节点数：1
#SBATCH -n 8                    ### CPU 核心数：8 (用于数据加速)
#SBATCH --gres=gpu:1            ### 显卡：1 块 NVIDIA A30
#SBATCH --time=48:00:00         ### 时间限制：48 小时 (400 Epoch 足够了)
#SBATCH -o logs/base_%j.out     ### 标准输出日志 (%j 为作业号)

# [Nety 报告] 正在初始化环境...
echo "[Nety] 任务启动时间: $(date)"

# 1. 自动创建输出和日志目录
mkdir -p logs
mkdir -p output/ablation_base_run1

# 2. 激活你的专属 Miniconda 环境
source /share/home/u2515063027/miniconda3/etc/profile.d/conda.sh
conda activate netmamba

# 3. 切换到项目根目录
cd /share/home/u2515063027/NetMamba/NetMamba-main

# 4. 执行 Baseline 训练
# 注意：这里调用的是备份修改后的逻辑文件 src/fine-tune_ablation_base.py
python src/fine-tune_ablation_base.py \
    --model net_mamba_classifier \
    --batch_size 128 \
    --epochs 400 \
    --output_dir output/ablation_base_run1 \
    --log_dir output/ablation_base_run1 \
    --num_workers 8 \
    --pin_mem

echo "[Nety] 任务结束时间: $(date)"
