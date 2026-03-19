import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# [Nety 核心修复：环境变量路由配置]
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
sys.path.append(SRC_DIR)

from src.models_net_mamba import net_mamba_classifier
from dataset.dataset_common import MultimodalTrafficDataset, USTC_20_CLASSES

# 反转字典，用于图表标签
IDX_TO_CLASS = {v: k for k, v in USTC_20_CLASSES.items()}
CLASS_NAMES = [IDX_TO_CLASS[i] for i in range(20)]


def plot_confusion_matrix(cm, output_path="confusion_matrix.png"):
    """
    [Nety 绘图引擎]：生成论文级别的混淆矩阵高分辨率图像
    """
    plt.figure(figsize=(16, 14))
    sns.set_theme(font_scale=1.2)
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                     cbar_kws={'label': 'Number of Flows'})

    plt.title('Hybrid NetMamba Confusion Matrix on USTC-TFC2016', fontsize=18, pad=20)
    plt.xlabel('Predicted Label', fontsize=16, labelpad=15)
    plt.ylabel('True Label', fontsize=16, labelpad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Nety 报告] 混淆矩阵已成功渲染并保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="[Nety] Hybrid NetMamba 核心效能评估引擎")
    parser.add_argument("--weight", type=str, default="output/hybrid_netmamba_run1/checkpoint-399.pth",
                        help="模型权重文件路径")
    parser.add_argument("--batch_size", type=int, default=64, help="测试批量大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    args = parser.parse_args()

    print(f"\\n[Nety 报告] 正在启动全局效能评估... 使用设备: {args.device.upper()}")

    # 1. 挂载测试集 (严格对齐 8:1:1 划分中的 Test 分支)
    print("[Nety 报告] 正在挂载 USTC-TFC2016 测试集...")
    test_dataset = MultimodalTrafficDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    # 2. 实例化模型并加载 400 Epoch 权重
    print(f"[Nety 报告] 正在加载模型权重: {args.weight}")
    model = net_mamba_classifier(num_classes=20)
    checkpoint = torch.load(args.weight, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(args.device)
    model.eval()

    all_preds = []
    all_targets = []
    total_samples = 0

    # 用于计算吞吐量的时间戳
    start_time = time.time()

    # 3. 批量前向推理
    print("[Nety 报告] 正在进行高吞吐量多模态特征融合与推理...")
    with torch.no_grad():
        for batch_idx, (imgs, pl, iat, targets) in enumerate(test_loader):
            imgs = imgs.to(args.device)
            pl = pl.to(args.device)
            iat = iat.to(args.device)
            targets = targets.to(args.device)

            # 核心融合前向传播
            logits = model(imgs, pl=pl, iat=iat)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            total_samples += targets.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(f"  - 已处理 {total_samples} 条流量样本...")

    end_time = time.time()

    # 4. [Nety 独家多分类严格计算引擎] 指标计算与科学定性
    total_time = end_time - start_time
    throughput = total_samples / total_time

    # 先生成标准的 20x20 混淆矩阵
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(20)))

    # 提取多分类的 True Positive, False Positive, False Negative, True Negative
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # 过滤掉支持度为 0 的"幽灵类别"（例如本测试集中的 Facetime 和 Nshisenz）
    support = cm.sum(axis=1)
    valid_mask = support > 0  # 只有真实存在的类别才参与宏平均计算 (动态分母)

    # 计算逐类别指标 (使用 np.divide 安全处理除零操作)
    precision = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
    recall = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision),
                   where=(precision + recall) != 0)
    fpr = np.divide(FP, FP + TN, out=np.zeros_like(FP, dtype=float), where=(FP + TN) != 0)
    fnr = np.divide(FN, FN + TP, out=np.zeros_like(FN, dtype=float), where=(FN + TP) != 0)  # FNR 本质上等于 1 - Recall

    # 计算严格的宏平均 (Macro Averages) - 只对存在样本的类别进行平均！
    macro_precision = np.mean(precision[valid_mask])
    macro_recall = np.mean(recall[valid_mask])
    macro_f1 = np.mean(f1[valid_mask])
    macro_fpr = np.mean(fpr[valid_mask])
    macro_fnr = np.mean(fnr[valid_mask])
    global_acc = accuracy_score(all_targets, all_preds)

    # 打印完美对齐的学术报告
    print("\n" + "=" * 85)
    print(" 🛡️ [Nety 核心效能诊断与学术报告] (完全体终极版)")
    print("=" * 85)
    print(f"测试集总规模: {total_samples} Flows")
    print(f"推理总耗时:   {total_time:.2f} 秒")
    print(f"动态吞吐量:   {throughput:.2f} Flows/Second (线速评估)")
    print("-" * 85)
    print(f"Global Accuracy  : {global_acc * 100:.4f}%")
    print(f"Macro Precision  : {macro_precision * 100:.4f}%")
    print(f"Macro Recall     : {macro_recall * 100:.4f}%")
    print(f"Macro F1-Score   : {macro_f1 * 100:.4f}%")
    print(f"Macro FPR (误报) : {macro_fpr * 100:.4f}% (越低越好)")
    print(f"Macro FNR (漏报) : {macro_fnr * 100:.4f}% (越低越好)")
    print("-" * 85)

    # 打印包含 FPR 和 FNR 的细粒度表格
    print(
        f"{'Class Name':<16} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'FPR':<8} | {'FNR':<8} | {'Support'}")
    print("-" * 85)
    for i in range(20):
        if valid_mask[i]:
            print(
                f"{CLASS_NAMES[i]:<16} | {precision[i]:.4f}    | {recall[i]:.4f}   | {f1[i]:.4f}   | {fpr[i]:.4f}   | {fnr[i]:.4f}   | {support[i]}")
        else:
            print(
                f"{CLASS_NAMES[i]:<16} | {'-':<9} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<8} | {support[i]} (集内缺失)")
    print("=" * 85)

    # 5. 绘制并保存混淆矩阵
    plot_confusion_matrix(cm, output_path="confusion_matrix.png")


if __name__ == "__main__":
    main()