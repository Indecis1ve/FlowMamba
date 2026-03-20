import os, glob, numpy as np, torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

FLOWS_ROOT = "/share/home/u2515063027/NetMamba/NetMamba-main/data/USTC_Flows_Processed"
METADATA_ROOT = "/share/home/u2515063027/NetMamba/NetMamba-main/data/USTC_Metadata"
USTC_20_CLASSES = {"BitTorrent": 0, "Facetime": 1, "FTP": 2, "Gmail": 3, "MySQL": 4, "Outlook": 5, "Skype": 6, "SMB": 7, "Weibo": 8, "WorldOfWarcraft": 9, "Cridex": 10, "Geodo": 11, "Htbot": 12, "Miuref": 13, "Neris": 14, "Nshisenz": 15, "Shifu": 16, "Tinba": 17, "Virut": 18, "Zeus": 19}

def fast_read_pcap_bytes(pcap_path, max_bytes=1600):
    try:
        with open(pcap_path, 'rb') as f:
            f.read(24)
            data = bytearray()
            while len(data) < max_bytes:
                pkt_hdr = f.read(16); 
                if not pkt_hdr: break
                incl_len = int.from_bytes(pkt_hdr[8:12], byteorder='little')
                data.extend(f.read(incl_len))
        data = data[:max_bytes]
        if len(data) < max_bytes: data.extend(b'\x00' * (max_bytes - len(data)))
        arr = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 255.0) * 2.0 - 1.0
        return torch.tensor(arr.reshape(1, 40, 40), dtype=torch.float32)
    except: return torch.zeros((1, 40, 40), dtype=torch.float32)

class MultimodalTrafficDataset(Dataset):
    def __init__(self, data_path=None, split="train", seed=42):
        self.samples = []
        all_sessions, labels = [], []
        self.stats_cache = {}
        
        npy_files = glob.glob(os.path.join(METADATA_ROOT, "*_stats.npy"))
        for npy_file in npy_files:
            try: stats_data = np.load(npy_file, allow_pickle=True).item()
            except: continue
            class_name = os.path.basename(npy_file).split('-')[1]
            if class_name not in USTC_20_CLASSES: continue
            class_id = USTC_20_CLASSES[class_name]
            
            for sf_name in stats_data.keys():
                pcap_path = os.path.join(FLOWS_ROOT, class_name, sf_name)
                if os.path.exists(pcap_path):
                    self.stats_cache[sf_name] = stats_data[sf_name]
                    all_sessions.append({"pcap": pcap_path, "sf_name": sf_name, "label": class_id})
                    labels.append(class_id)
                    
        train_val, test, y_train_val, _ = train_test_split(all_sessions, labels, test_size=0.1, random_state=seed, stratify=labels)
        train, valid, _, _ = train_test_split(train_val, y_train_val, test_size=0.1111, random_state=seed, stratify=y_train_val)
        self.samples = {"train": train, "valid": valid, "test": test}[split]

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        imgs = fast_read_pcap_bytes(item["pcap"])
        
        stat = self.stats_cache.get(item["sf_name"], {"pl": [0]*100, "iat": [0.0]*100})
        pl_data = stat.get("pl", [])[:100]
        iat_data = stat.get("iat", [])[:100]
        
        pl_data += [0] * (100 - len(pl_data))
        iat_data += [0.0] * (100 - len(iat_data))
        
        # [Nety 核心护甲：归一化处理]
        pl_tensor = torch.nan_to_num(torch.tensor(pl_data, dtype=torch.float32) / 1500.0)
        
        # [Nety 绝杀 Bug 的武器：对数压缩，防止数值爆炸撑破 Float16！]
        iat_raw = torch.tensor(iat_data, dtype=torch.float32)
        iat_raw = torch.clamp(iat_raw, min=0.0) # 剔除罕见的负数时间异常
        iat_tensor = torch.nan_to_num(torch.log1p(iat_raw)) 
        
        return imgs, pl_tensor, iat_tensor, item["label"]
