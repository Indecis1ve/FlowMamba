import os
import glob
import numpy as np

# 路径对齐
METADATA_ROOT = "/share/home/u2515063027/NetMamba/NetMamba-main/data/USTC_Metadata"

# 20类学术划分
BENIGN_CLASSES = ["BitTorrent", "Facetime", "FTP", "Gmail", "MySQL", "Outlook", "Skype", "SMB", "Weibo", "WorldOfWarcraft"]
MALWARE_CLASSES = ["Cridex", "Geodo", "Htbot", "Miuref", "Neris", "Nshisenz", "Shifu", "Tinba", "Virut", "Zeus"]

class_stats = []
total_flows = 0

npy_files = glob.glob(os.path.join(METADATA_ROOT, "*_stats.npy"))

# 遍历扫描
for npy_file in npy_files:
    data = np.load(npy_file, allow_pickle=True).item()
    class_name = os.path.basename(npy_file).split('-')[1]
    count = len(data)
    group = "Benign" if class_name in BENIGN_CLASSES else "Malicious"
    class_stats.append({"name": class_name, "count": count, "group": group})
    total_flows += count

# 排序输出：先排阵营，再排字母顺序
class_stats.sort(key=lambda x: (x['group'], x['name']))

print(f"\n{'Group':<12} | {'Category':<18} | {'Count':<8} | {'Percentage':<10}")
print("-" * 55)
for s in class_stats:
    perc = (s['count'] / total_flows) * 100
    print(f"{s['group']:<12} | {s['name']:<18} | {s['count']:<8} | {perc:>8.2f}%")

print("-" * 55)
print(f"Total Flows: {total_flows}")