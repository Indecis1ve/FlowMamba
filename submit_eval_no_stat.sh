#!/bin/bash
#SBATCH -J netmamba_eval_no_stat        
#SBATCH -p gpu                  
#SBATCH -N 1                    
#SBATCH -n 8                    
#SBATCH --gres=gpu:1            
#SBATCH --time=00:15:00         
#SBATCH -o logs/eval_%j.out     

echo "[Nety 报告] 正在初始化 SLURM 评估节点环境变量..."
mkdir -p logs

# ==========================================
# [Nety 核心修复：指向你私人的 Miniconda 路径]
# ==========================================
source /share/home/u2515063027/miniconda3/etc/profile.d/conda.sh

# 纯净激活你的专属环境
conda activate netmamba

# 切换到项目绝对工作目录
cd /share/home/u2515063027/NetMamba/NetMamba-main

echo "[Nety 报告] 环境就绪，A30 算力全开，正在启动大批量时空多模态特征评估..."

# 执行高吞吐量评估
python evaluate_ablation_no_stat.py \
    --weight output/hybrid_netmamba_run1/checkpoint-399.pth \
    --batch_size 128 \
    --num_workers 8

echo "[Nety 报告] 全局测试集评估任务执行完毕，请查收混淆矩阵与性能指标！"