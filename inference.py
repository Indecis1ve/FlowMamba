import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scapy.all import rdpcap


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
sys.path.append(SRC_DIR)


from src.models_net_mamba import net_mamba_classifier

# =========================================
# 1. 分类标签逆映射 (Label Mapping)
# =========================================
USTC_20_CLASSES = {
    "BitTorrent": 0, "Facetime": 1, "FTP": 2, "Gmail": 3, "MySQL": 4,
    "Outlook": 5, "Skype": 6, "SMB": 7, "Weibo": 8, "WorldOfWarcraft": 9,
    "Cridex": 10, "Geodo": 11, "Htbot": 12, "Miuref": 13, "Neris": 14,
    "Nshisenz": 15, "Shifu": 16, "Tinba": 17, "Virut": 18, "Zeus": 19
}
# 反转字典，用于将模型输出的数字索引转换为人类可读的字符串
IDX_TO_CLASS = {v: k for k, v in USTC_20_CLASSES.items()}


# =========================================
# 2. 实战 PCAP 多模态特征提取与严格预处理
# =========================================
def extract_live_pcap_features(pcap_path, max_bytes=1600, pl_seq_len=10, iat_seq_len=9):
    """
    [Nety 核心解析管线]：从原始 PCAP 提取空间与统计特征，严格对齐训练时的归一化标准。
    """
    packets = rdpcap(pcap_path)
    if len(packets) == 0:
        raise ValueError("[Nety 警报] 提供的 PCAP 文件为空！")

    # --- 模态 A: 空间字节序列 (1D-CNN 输入) ---
    raw_bytes = bytearray()
    for pkt in packets:
        raw_bytes.extend(bytes(pkt))
        if len(raw_bytes) >= max_bytes:
            break

    # 截断或补零至 1600 字节
    raw_bytes = raw_bytes[:max_bytes]
    if len(raw_bytes) < max_bytes:
        raw_bytes.extend(b'\x00' * (max_bytes - len(raw_bytes)))

    # [严格对齐]：(arr / 255.0) * 2.0 - 1.0 并 Reshape 为 (1, 40, 40)
    arr_np = np.array(raw_bytes, dtype=np.float32)
    arr_np = (arr_np / 255.0) * 2.0 - 1.0
    img_tensor = torch.tensor(arr_np).reshape(1, 1, 40, 40)  # 增加 Batch 维度

    # --- 模态 B: 时空统计特征 PL (Packet Length) 和 IAT (Inter-Arrival Time) ---
    pl_list = []
    time_list = []

    # 提取前 N 个包的长度和到达时间
    for i, pkt in enumerate(packets):
        if i >= pl_seq_len:
            break
        pl_list.append(len(pkt))
        time_list.append(float(pkt.time))

    # 补齐不足的包
    while len(pl_list) < pl_seq_len:
        pl_list.append(0)
        time_list.append(time_list[-1] if time_list else 0.0)

    # 计算 IAT (前一项与后一项的时间差)
    iat_list = [time_list[i + 1] - time_list[i] for i in range(iat_seq_len)]

    # [严格对齐]：PL 除以 1500.0，IAT 经过 log1p 处理
    pl_tensor = torch.tensor(pl_list, dtype=torch.float32) / 1500.0
    iat_tensor = torch.log1p(torch.tensor(iat_list, dtype=torch.float32))

    # 增加 Batch 维度
    pl_tensor = pl_tensor.unsqueeze(0)
    iat_tensor = iat_tensor.unsqueeze(0)

    return img_tensor, pl_tensor, iat_tensor


# =========================================
# 3. 主干推理逻辑
# =========================================
def main():
    parser = argparse.ArgumentParser(description="[Nety] Hybrid NetMamba 异常流量检测实战推理引擎")
    parser.add_argument("--pcap", type=str, required=True, help="待检测的单条 .pcap 文件路径")
    parser.add_argument("--weight", type=str, default="output/hybrid_netmamba_run1/checkpoint-399.pth",
                        help="模型权重文件路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="推理计算设备")
    args = parser.parse_args()

    print(f"\\n[Nety 报告] 正在启动监测节点... 使用设备: {args.device.upper()}")

    # 1. 实例化轻量级 Mamba 模型并加载权重
    print(f"[Nety 报告] 正在加载 400 Epoch 训练权重: {args.weight}")
    model = net_mamba_classifier(num_classes=20)

    try:
        checkpoint = torch.load(args.weight, map_location="cpu")
        # 兼容不同保存格式（如果保存了整个模型结构或是 state_dict）
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"[Nety 错误] 权重加载失败，请检查路径。详细错误: {e}")
        return

    model.to(args.device)
    model.eval()

    # 2. 提取多模态特征
    print(f"[Nety 报告] 正在实时解析流量包: {args.pcap}")
    try:
        img_tensor, pl_tensor, iat_tensor = extract_live_pcap_features(args.pcap)
    except Exception as e:
        print(f"[Nety 错误] 数据包解析失败。详细错误: {e}")
        return

    img_tensor = img_tensor.to(args.device)
    pl_tensor = pl_tensor.to(args.device)
    iat_tensor = iat_tensor.to(args.device)

    # 3. 模型前向推理
    print("[Nety 报告] 正在进行时空特征融合与威胁分析...")
    with torch.no_grad():
        # 传入四元组进行前向传播
        logits = model(img_tensor, pl_tensor, iat_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze()

        # 获取置信度最高的前 3 名
        top3_prob, top3_idx = torch.topk(probabilities, 3)

    # 4. 打印安全诊断报告
    predicted_class = IDX_TO_CLASS[top3_idx[0].item()]
    is_malware = predicted_class in ["Cridex", "Geodo", "Htbot", "Miuref", "Neris", "Nshisenz", "Shifu", "Tinba",
                                     "Virut", "Zeus"]

    print("\\n" + "=" * 50)
    print(" 🛡️ [Nety 流量异常检测报告]")
    print("=" * 50)
    print(f"检测目标: {args.pcap}")
    print(f"安全定性: {'🚨 恶意流量 (Malware)' if is_malware else '✅ 正常流量 (Benign)'}")
    print(f"主要分类: {predicted_class} (置信度: {top3_prob[0].item() * 100:.2f}%)")
    print("-" * 50)
    print("Top-3 可能性预测:")
    for i in range(3):
        cls_name = IDX_TO_CLASS[top3_idx[i].item()]
        prob = top3_prob[i].item() * 100
        print(f" {i + 1}. {cls_name:<15} : {prob:.2f}%")
    print("=" * 50 + "\\n")


if __name__ == "__main__":
    main()