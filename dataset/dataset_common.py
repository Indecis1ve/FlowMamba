import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import logging

# ==========================================
# [Nety 路径配置]：硬编码指向处理后的流量与元数据
# ==========================================
FLOWS_ROOT = "/share/home/u2515063027/NetMamba/NetMamba-main/data/USTC_Flows_Processed"
METADATA_ROOT = "/share/home/u2515063027/NetMamba/NetMamba-main/data/USTC_Metadata"

# [Nety 修复] 引入 20 分类字典，用于从文件名反推标签
USTC_20_CLASSES = {
    "BitTorrent": 0, "Facetime": 1, "FTP": 2, "Gmail": 3, "MySQL": 4,
    "Outlook": 5, "Skype": 6, "SMB": 7, "Weibo": 8, "WorldOfWarcraft": 9,
    "Cridex": 10, "Geodo": 11, "Htbot": 12, "Miuref": 13, "Neris": 14,
    "Nshisenz": 15, "Shifu": 16, "Tinba": 17, "Virut": 18, "Zeus": 19
}

def fast_read_pcap_bytes(pcap_path, max_bytes=1600):
    """
    [Nety 极速核心]：直接读取原始二进制字节流，转化为 40x40 图像张量。
    """
    try:
        with open(pcap_path, 'rb') as f:
            f.read(24)  # 跳过 PCAP 全局头
            data = bytearray()
            while len(data) < max_bytes:
                pkt_hdr = f.read(16)
                if not pkt_hdr: break
                incl_len = int.from_bytes(pkt_hdr[8:12], byteorder='little')
                data.extend(f.read(incl_len))

        data = data[:max_bytes]
        if len(data) < max_bytes:
            data.extend(b'\x00' * (max_bytes - len(data)))

        # 映射到 [-1, 1] 范围，适配 Mamba 预训练权重的分布
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        arr = (arr / 255.0) * 2.0 - 1.0
        arr = arr.reshape(1, 40, 40)
        return torch.tensor(arr, dtype=torch.float32)
    except Exception as e:
        # 异常静默填充，防止训练崩溃
        return torch.zeros((1, 40, 40), dtype=torch.float32)


class MultimodalTrafficDataset(Dataset):
    def __init__(self, data_path=None, split="train", seed=42):
        self.samples = []
        all_sessions = []
        labels = []

        npy_files = glob.glob(os.path.join(METADATA_ROOT, "*_stats.npy"))
        print(f"[Nety] 正在扫描元数据并构建 {split} 集索引...")
        
        for npy_file in npy_files:
            try:
                stats_data = np.load(npy_file, allow_pickle=True).item()
            except Exception as e:
                continue

          
            # 示例文件名: Benign-BitTorrent-BitTorrent_stats.npy
            basename = os.path.basename(npy_file)
            parts = basename.split('-')
            
            if len(parts) >= 2:
                class_name = parts[1]  # 提取第二段，即 class_name
            else:
                continue
            
            # 如果解析出的类别名不在我们的 20 分类字典中，则跳过
            if class_name not in USTC_20_CLASSES:
                continue
            
            class_id = USTC_20_CLASSES[class_name]

            for sf_name, meta in stats_data.items():
                pcap_path = os.path.join(FLOWS_ROOT, class_name, sf_name)

                # 确保对应的 PCAP 文件真实存在
                if os.path.exists(pcap_path):
                    all_sessions.append({
                        "pcap": pcap_path,
                        "pl": torch.tensor(meta["pl"], dtype=torch.float32),
                        "iat": torch.tensor(meta["iat"], dtype=torch.float32),
                        "label": class_id
                    })
                    labels.append(class_id)

        if not all_sessions:
            print("[Nety 警告] 未加载到有效样本，请检查 FLOWS_ROOT 路径下的 PCAP 文件是否存在！")
            return

        # 严格执行 8:1:1 的流级别划分 (Stratified 按照标签分层采样保证类别平衡)
        train_val, test, y_train_val, _ = train_test_split(
            all_sessions, labels, test_size=0.1, random_state=seed, stratify=labels
        )
        train, valid, _, _ = train_test_split(
            train_val, y_train_val, test_size=0.1111, random_state=seed, stratify=y_train_val
        )

        if split == "train":
            self.samples = train
        elif split == "valid":
            self.samples = valid
        else:
            self.samples = test

        print(f"[Nety 报告] {split} 集构建完成，规模: {len(self.samples)} 条。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # 1. 空间特征 (Image: 已经归一化过)
        imgs_np = fast_read_pcap_bytes(item["pcap"])
        imgs_tensor = imgs_np.clone().detach().reshape(1, 40, 40)

        # ==================== [Nety 核心修复：多模态特征缩放] ====================
        # 2. 时空特征 (PL & IAT) 标准化处理
        # PL (报文长度) 最大通常为 1500 字节，我们使用简单的最大值除法将其缩放到 [0, 1] 左右
        pl_tensor = item["pl"] / 1500.0 
        
        # IAT (时间间隔) 分布极不均匀（长尾分布），使用 log1p (即 ln(1+x)) 可以极好地平滑极端值
        iat_tensor = torch.log1p(item["iat"])
        # =========================================================================

        # 3. 标签
        target = item["label"]

        return imgs_tensor, pl_tensor, iat_tensor, target