#!/bin/bash
#SBATCH -J NetMamba_Benchmark    ### 作业名称
#SBATCH -p gpu                   ### 提交到 gpu 队列
#SBATCH -N 1                     ### 申请 1 个节点
#SBATCH --gres=gpu:1             ### 申请 1 块 GPU (NVIDIA A30)
#SBATCH --cpus-per-task=4        ### 申请 4 个 CPU 核心 (保证 CPU 吞吐量测试不卡顿)
#SBATCH --mem=32G                ### 申请 32GB 内存
#SBATCH --time="01:00:00"        ### 基准测试很快，申请 1 小时足够，有助于快速排队
#SBATCH -o logs/%j_bench.out     ### 性能测试结果输出
#SBATCH -e logs/%j_bench.err     ### 错误日志

# ==========================================
# 阶段一：环境初始化
# ==========================================
echo "[Nety 监测台] 正在准备基准测试环境..."
source /share/home/u2515063027/miniconda3/etc/profile.d/conda.sh
conda activate netmamba

# [核心路径配置] 确保脚本能找到 src 下的模块
export PYTHONPATH="/share/home/u2515063027/NetMamba/NetMamba-main:/share/home/u2515063027/NetMamba/NetMamba-main/src:${PYTHONPATH}"

# 算子编译环境 (Benchmark 同样需要加载自定义算子)
export MAX_JOBS=4
export CUDA_HOME=$CONDA_PREFIX
export MAMBA_FORCE_BUILD="TRUE"

# ==========================================
# 阶段二：启动性能评估
# ==========================================
PROJECT_DIR="/share/home/u2515063027/NetMamba/NetMamba-main"
cd ${PROJECT_DIR}

# 确保日志目录存在
mkdir -p logs

echo "[Nety 监测台] 正在执行模型性能全维度分析..."

# 运行我们之前设计的 benchmark 脚本
python src/benchmark_nety.py

echo "[Nety 监测台] 测试圆满结束！请查看 logs/ 目录下的输出文件。"