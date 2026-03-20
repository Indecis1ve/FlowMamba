#!/bin/bash
#SBATCH -J mem_test   ###指定作业名称
#SBATCH -p gpu   ####指定队列名称cpu/gpu/fat
#SBATCH -N 1  ###指定主机数量
#SBATCH -n 8  ###指定核心数
#SBATCH --gres=gpu:1  ##指定GPU卡数量,需要Gpu才写
#SBATCH -o logs/measure_mem_%j.out

source /share/home/u2515063027/miniconda3/etc/profile.d/conda.sh  ##加载conda环境变量
module load cuda-11.7 ##加载cuda 可以使用module ava查看已安装版本
conda activate netmamba  ##进入自己的环境 

python /share/home/u2515063027/NetMamba/NetMamba-main/measure_memory.py   ###执行pytorch命令
