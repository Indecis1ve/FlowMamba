import torch
import torch.nn as nn
import gc

try:
    from mamba_ssm import Mamba
except ImportError:
    print("❌ 找不到 mamba_ssm！")
    exit()

# --- 1. 定义我们 NetMamba 的核心主干 (4层 Mamba) ---
class NetMambaCore(nn.Module):
    def __init__(self, d_model=128, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) 
            for _ in range(n_layers)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# --- 2. 定义原生的 Vanilla Transformer (4层，禁用内存优化) ---
class VanillaTransformerCore(nn.Module):
    def __init__(self, d_model=128, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=d_model*4, batch_first=True)
            for _ in range(n_layers)
        ])
    def forward(self, x):
        # [Nety 核心护盾]：强制禁用底层 FlashAttention 和 Memory Efficient Attention
        # 暴露出 Transformer 真实的 O(L^2) 平方级内存瓶颈！
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            for layer in self.layers:
                x = layer(x)
        return x

def measure_peak_memory(model, batch_size, seq_len, d_model, device="cuda"):
    model.to(device)
    model.train()
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    
    try:
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        del x, out, loss
        model.zero_grad()
        return round(peak_mem, 2)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return "OOM (显存爆炸)"
        else:
            raise e
    finally:
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    BATCH_SIZE = 2
    D_MODEL = 128
    seq_lengths = list(range(1000, 10001, 1000))
    
    results_transformer = []
    results_mamba = []

    print(f"\n📏 正在评估 GPU 显存 (Batch Size={BATCH_SIZE}, Layers=4)")
    print("-" * 75)
    print(f"{'序列长度 (L)':<15} | {'Vanilla Transformer (MB)':<25} | {'NetMamba 主干 (MB)':<20}")
    print("-" * 75)

    for L in seq_lengths:
        # 测试原生 Transformer
        tf_model = VanillaTransformerCore(d_model=D_MODEL, n_layers=4)
        mem_tf = measure_peak_memory(tf_model, BATCH_SIZE, L, D_MODEL, device)
        results_transformer.append(mem_tf)
        del tf_model

        # 测试我们的 NetMamba
        mamba_model = NetMambaCore(d_model=D_MODEL, n_layers=4)
        mem_mamba = measure_peak_memory(mamba_model, BATCH_SIZE, L, D_MODEL, device)
        results_mamba.append(mem_mamba)
        del mamba_model

        print(f"{L:<15} | {str(mem_tf):<25} | {str(mem_mamba):<20}")

    print("-" * 75)
    print("\n📊 可直接用于画图的坐标轴数组：")
    print(f"X_lengths = {seq_lengths}")
    print(f"Y_Transformer = {results_transformer}")
    print(f"Y_NetMamba = {results_mamba}")
