import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# [Nety 核心配置]
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
sys.path.append(SRC_DIR)

from src.models_net_mamba import net_mamba_classifier
from dataset.dataset_common import MultimodalTrafficDataset, USTC_20_CLASSES

IDX_TO_CLASS = {v: k for k, v in USTC_20_CLASSES.items()}
CLASS_NAMES = [IDX_TO_CLASS[i] for i in range(20)]


def plot_confusion_matrix(cm, output_path="cm_ablation_no_stat.png"):
    plt.figure(figsize=(16, 14))
    sns.set_theme(font_scale=1.2)
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',  # 使用橙色调区分消融实验
                     xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                     cbar_kws={'label': 'Number of Flows'})
    plt.title('Ablation Study: FlowMamba (w/o PL & IAT) on USTC-TFC2016', fontsize=18, pad=20)
    plt.xlabel('Predicted Label', fontsize=16, labelpad=15)
    plt.ylabel('True Label', fontsize=16, labelpad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Nety 报告] 消融混淆矩阵已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="[Nety] FlowMamba 极速消融实验引擎 - 屏蔽 PL/IAT")
    parser.add_argument("--weight", type=str, default="output/hybrid_netmamba_run1/checkpoint-399.pth")
    parser.add_argument("--batch_size", type=int, default=128)  # A30 显存充足，建议 128
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"\n[Nety 报告] 启动消融评估：Variant 2 (1D-CNN + Mamba, 无统计特征)...")

    # 1. 加载测试集
    test_dataset = MultimodalTrafficDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    # 2. 模型加载
    model = net_mamba_classifier(num_classes=20)
    checkpoint = torch.load(args.weight, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(args.device)
    model.eval()

    all_preds, all_targets = [], []
    total_samples = 0
    start_time = time.time()

    # 3. 屏蔽推理循环
    print("[Nety 报告] 正在模拟“统计特征缺失”场景进行推理...")
    with torch.no_grad():
        for imgs, pl, iat, targets in test_loader:
            imgs = imgs.to(args.device)
            targets = targets.to(args.device)

            # ====================================================
            # [Nety 极速消融补丁]：强行屏蔽 PL 和 IAT
            # 即使模型加载了完全体权重，我们也只给它全 0 的统计特征
            # ====================================================
            pl_masked = torch.zeros_like(pl).to(args.device)
            iat_masked = torch.zeros_like(iat).to(args.device)

            logits = model(imgs, pl=pl_masked, iat=iat_masked)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            total_samples += targets.size(0)

    # 4. 指标计算
    total_time = time.time() - start_time
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(20)))

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    support = cm.sum(axis=1)
    valid_mask = support > 0

    precision = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
    recall = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision),
                   where=(precision + recall) != 0)
    fpr = np.divide(FP, FP + TN, out=np.zeros_like(FP, dtype=float), where=(FP + TN) != 0)
    fnr = np.divide(FN, FN + TP, out=np.zeros_like(FN, dtype=float), where=(FN + TP) != 0)

    print("\n" + "=" * 85)
    print(" 🛡️ [Nety 消融实验报告：Variant 2 - 无多模态特征]")
    print("=" * 85)
    print(f"Global Accuracy  : {accuracy_score(all_targets, all_preds) * 100:.4f}%")
    print(f"Macro F1-Score   : {np.mean(f1[valid_mask]) * 100:.4f}%")
    print(f"Macro FPR (误报) : {np.mean(fpr[valid_mask]) * 100:.4f}%")
    print(f"Macro FNR (漏报) : {np.mean(fnr[valid_mask]) * 100:.4f}%")
    print("-" * 85)
    print(f"{'Class Name':<16} | {'Precision':<9} | {'F1-Score':<8} | {'FNR (漏报)':<8}")
    for i in range(20):
        if valid_mask[i]:
            print(f"{CLASS_NAMES[i]:<16} | {precision[i]:.4f}    | {f1[i]:.4f}   | {fnr[i]:.4f}")
    print("=" * 85)

    plot_confusion_matrix(cm)


if __name__ == "__main__":
    main()