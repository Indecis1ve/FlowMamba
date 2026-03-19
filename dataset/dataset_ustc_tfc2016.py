import os
import subprocess
import numpy as np
from tqdm import tqdm
from scapy.all import rdpcap
import logging

# 屏蔽 Scapy 的警告信息，保持终端整洁
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

# ==========================================
# [Nety 配置台]：请确保以下路径在 HPC 上正确
# ==========================================
SPLITTER_BIN = "/share/home/u2515063027/NetMamba/NetMamba-main/util/pcap_tool/splitter"
BASE_DIR = "/share/home/u2515063027/NetMamba/NetMamba-main/data/USTC-TFC2016_RAW/USTC-TFC2016-master"
OUTPUT_FLOW_ROOT = "/share/home/u2515063027/NetMamba/NetMamba-main/data/USTC_Flows_Processed"
METADATA_ROOT = "/share/home/u2515063027/NetMamba/NetMamba-main/data/USTC_Metadata"

# 多模态特征维度定义 (根据魔改项目需求)
MAX_PKTS = 10  # 提取前10个包
PL_DIM = 10  # 报文长度序列维度
IAT_DIM = 9  # 到达时间间隔序列维度 (10个包产生9个间隔)

# 20 分类字典对齐
USTC_20_CLASSES = {
    # Benign (0-9)
    "BitTorrent": 0, "Facetime": 1, "FTP": 2, "Gmail": 3, "MySQL": 4,
    "Outlook": 5, "Skype": 6, "SMB": 7, "Weibo": 8, "WorldOfWarcraft": 9,
    # Malware (10-19)
    "Cridex": 10, "Geodo": 11, "Htbot": 12, "Miuref": 13, "Neris": 14,
    "Nshisenz": 15, "Shifu": 16, "Tinba": 17, "Virut": 18, "Zeus": 19
}


def extract_metadata_logic(pcap_path):
    """
    [Nety 核心提取引擎]：从流 PCAP 中提取 PL 和 IAT 序列
    """
    try:
        pkts = rdpcap(pcap_path)
        if len(pkts) < 2: return None  # 过滤掉只有1个包或空的流

        # 1. 提取 Packet Lengths (PL)
        pl_seq = [len(p) for p in pkts[:MAX_PKTS]]
        # Padding
        if len(pl_seq) < PL_DIM:
            pl_seq += [0] * (PL_DIM - len(pl_seq))

        # 2. 提取 Inter-Arrival Times (IAT)
        iat_seq = []
        for i in range(min(len(pkts), MAX_PKTS) - 1):
            diff = pkts[i + 1].time - pkts[i].time
            iat_seq.append(float(diff))

        # Padding
        if len(iat_seq) < IAT_DIM:
            iat_seq += [0.0] * (IAT_DIM - len(iat_seq))

        return {"pl": pl_seq[:PL_DIM], "iat": iat_seq[:IAT_DIM]}
    except Exception as e:
        return None


def main():
    os.makedirs(OUTPUT_FLOW_ROOT, exist_ok=True)
    os.makedirs(METADATA_ROOT, exist_ok=True)

    print(f"[Nety 启动] 正在扫描数据集: {BASE_DIR}")

    for category in ["Benign", "Malware"]:
        category_dir = os.path.join(BASE_DIR, category)
        if not os.path.exists(category_dir):
            print(f"[警告] 目录不存在: {category_dir}")
            continue

        for class_name in USTC_20_CLASSES.keys():
            # --- [Nety 核心修正：二段式路径探测] ---
            class_path_as_dir = os.path.join(category_dir, class_name)
            class_path_as_file = os.path.join(category_dir, class_name + ".pcap")

            pcap_files_to_process = []

            # 检查是否为文件夹 (如 Benign/BitTorrent/)
            if os.path.isdir(class_path_as_dir):
                pcap_files_to_process = [
                    os.path.join(class_path_as_dir, f)
                    for f in os.listdir(class_path_as_dir) if f.endswith(".pcap")
                ]
            # 检查是否为单个 PCAP 文件 (如 Benign/BitTorrent.pcap)
            elif os.path.isfile(class_path_as_file):
                pcap_files_to_process = [class_path_as_file]

            if not pcap_files_to_process:
                continue  # 当前类别不在此 category 目录下

            # 处理该类别下的所有原始 PCAP
            for pcap_full_path in tqdm(pcap_files_to_process, desc=f"处理 {category}/{class_name}"):
                pcap_file_name = os.path.basename(pcap_full_path)
                # 生成唯一前缀，防止重名覆盖
                flow_prefix = f"{category}-{class_name}-{pcap_file_name[:-5]}"

                # 检查是否已经提取过
                target_npy = os.path.join(METADATA_ROOT, f"{flow_prefix}_stats.npy")
                if os.path.exists(target_npy):
                    continue

                # 为切分后的流创建临时存储目录
                current_flow_dir = os.path.join(OUTPUT_FLOW_ROOT, class_name)
                os.makedirs(current_flow_dir, exist_ok=True)

                # 调用 Splitter 进行流切分
                # -p 指定前缀，-f five_tuple 按照五元组切分
                subprocess.run(
                    f"{SPLITTER_BIN} -i '{pcap_full_path}' -o '{current_flow_dir}' -p {flow_prefix}- -f five_tuple",
                    shell=True, capture_output=True
                )

                # 扫描生成的流文件并提取元数据
                stats_map = {}
                for sf in os.listdir(current_flow_dir):
                    if sf.startswith(flow_prefix) and sf.endswith(".pcap"):
                        res = extract_metadata_logic(os.path.join(current_flow_dir, sf))
                        if res:
                            stats_map[sf] = res  # 存储该流的 PL 和 IAT

                # 持久化存储元数据
                if stats_map:
                    np.save(target_npy, stats_map)

                # [可选]：为了节省存储空间，可以在此处删除生成的中间流文件
                # for sf in os.listdir(current_flow_dir):
                #     if sf.startswith(flow_prefix): os.remove(os.path.join(current_flow_dir, sf))

    print("[Nety 报告] 预处理完成！元数据已存放于:", METADATA_ROOT)


if __name__ == "__main__":
    main()