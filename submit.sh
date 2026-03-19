#!/bin/bash
#SBATCH -J Hybrid_NetMamba       
#SBATCH -p gpu                   
#SBATCH -N 1                     
#SBATCH --gres=gpu:1             
#SBATCH --cpus-per-task=4        ### [Nety 新增] 申请 4 个 CPU 核心
#SBATCH --mem=32G                ### [Nety 新增] 申请 32GB 的 CPU 内存，防止内存溢出被强杀！
#SBATCH --time="48:00:00"        
#SBATCH -o logs/%j_hybrid_run.out 
#SBATCH -e logs/%j_hybrid_run.err 

# ==========================================
# 阶段一：HPC 环境准备与算力对齐
# ==========================================
echo "[Nety 监测台] 正在初始化 HPC 计算节点..."
source /share/home/u2515063027/miniconda3/etc/profile.d/conda.sh
conda activate netmamba

# [Nety 全局路径] 
export PYTHONPATH="/share/home/u2515063027/NetMamba/NetMamba-main:/share/home/u2515063027/NetMamba/NetMamba-main/src:${PYTHONPATH}"

# [Nety DDP 通信基站] 
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# 防 OOM 与编译保护
export MAX_JOBS=4
export CUDA_HOME=$CONDA_PREFIX
export MAMBA_FORCE_BUILD="TRUE"

# ==========================================
# 阶段二：工作区检查与模型点火
# ==========================================
PROJECT_DIR="/share/home/u2515063027/NetMamba/NetMamba-main"
cd ${PROJECT_DIR}

mkdir -p logs
mkdir -p output/hybrid_netmamba_run1/tensorboard

echo "[Nety 监测台] 算力通道已就绪！开始启动多模态 NetMamba 微调任务..."

# [Nety 核心修改] 强行指定 --num_workers 为 2，降低 I/O 内存压力
python src/fine-tune.py \
    --epochs 400 \
    --nb_classes 20 \
    --lr 5e-4 \
    --warmup_epochs 5 \
    --batch_size 64 \
    --num_workers 2 \
    --output_dir ./output/hybrid_netmamba_run1 \
    --log_dir ./output/hybrid_netmamba_run1/tensorboard

echo "[Nety 监测台] 任务提交流程结束。"