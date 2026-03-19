#!/bin/bash
#SBATCH -J netmamba_infer       ### 指定作业名称为 netmamba_infer
#SBATCH -p gpu                  ### 指定队列名称为 gpu 
#SBATCH -N 1                    ### 指定主机数量为 1 
#SBATCH -n 4                    ### 指定核心数 (单条推理不需要太多 CPU 核心，4 个足够)
#SBATCH --gres=gpu:1            ### 申请 1 块 NVIDIA A30 GPU 卡 [cite: 7, 649]
#SBATCH --time=00:10:00         ### 指定运行时间 (单条 PCAP 推理极快，10分钟绰绰有余)
#SBATCH -o logs/infer_%j.out    ### 指定标准输出日志路径 (%j 会自动替换为 Job ID)

echo "[Nety 报告] 正在初始化 SLURM 节点环境变量..."

# 1. 加载 conda 环境变量
source /share/apps/anaconda3/etc/profile.d/conda.sh 

# 2. 激活 NetMamba 专属运行环境
conda activate netmamba [cite: 8, 649]

# 3. 切换到项目绝对路径，防止路径漂移
cd /share/home/u2515063027/NetMamba/NetMamba-main 

echo "[Nety 报告] 环境就绪，正在向 Mamba 模型下发实战 PCAP 检测指令..."

# 4. 执行我们在上一环节调试通过的推理命令
python inference.py \
    --pcap data/USTC_Flows_Processed/Cridex/Malware-Cridex-Cridex-62.75.184.70_8080_10.0.2.108_49158_TCP.pcap \
    --weight output/hybrid_netmamba_run1/checkpoint-399.pth

echo "[Nety 报告] 流量检测任务执行完毕！"