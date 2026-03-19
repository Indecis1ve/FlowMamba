#!/bin/bash
# [Nety 自动化备份引擎]：为 Baseline 消融实验创建受保护的副本

echo "[Nety 报告] 正在建立 Baseline 实验防御线..."

# 1. 备份模型定义文件
cp src/models_net_mamba.py src/models_net_mamba_ablation_base.py

# 2. 备份数据加载逻辑
cp dataset/dataset_common.py dataset/dataset_common_ablation_base.py

# 3. 备份训练微调脚本
cp src/fine-tune.py src/fine-tune_ablation_base.py

# 4. 备份训练执行引擎
cp src/engine.py src/engine_ablation_base.py

# 5. 备份提交脚本 (从你的目录树看，主脚本为 submit_train.sh)
cp submit_train.sh submit_ablation_base.sh

echo "----------------------------------------------------------"
echo "[Nety 报告] 备份完成！以下文件已就绪，等待注入 Baseline 逻辑："
echo " - src/models_net_mamba_ablation_base.py"
echo " - dataset/dataset_common_ablation_base.py"
echo " - src/fine-tune_ablation_base.py"
echo " - src/engine_ablation_base.py"
echo " - submit_ablation_base.sh"
echo "----------------------------------------------------------"
