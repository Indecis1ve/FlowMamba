import torch
import time
import numpy as np
from thop import profile
from models_net_mamba import net_mamba_classifier

def benchmark():
    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")
    
    # 1. 实例化模型 (20 分类)
    model = net_mamba_classifier(nb_classes=20).to(device_gpu)
    model.eval()

    # 2. 构造虚拟多模态输入 (Batch Size = 1)
    # 图像 [1, 1, 40, 40], PL [1, 10], IAT [1, 9]
    dummy_imgs = torch.randn(1, 1, 40, 40).to(device_gpu)
    dummy_pl = torch.randn(1, 10).to(device_gpu)
    dummy_iat = torch.randn(1, 9).to(device_gpu)

    print("="*30)
    print("[Nety 性能监测台] 开始评估...")

    # ======= 指标一：参数量 (Params) & 理论计算量 (FLOPs) =======
    # 注意：Mamba 算子的 FLOPs 统计可能需要 thop 兼容，这里提供估计值
    flops, params = profile(model, inputs=(dummy_imgs, dummy_pl, dummy_iat), verbose=False)
    print(f"🔹 Total Params: {params/1e6:.3f} M")
    print(f"🔹 Total FLOPs: {flops/1e6:.3f} MFLOPs")

    # ======= 指标二：GPU 显存占用 (GPU Memory) =======
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(dummy_imgs, dummy_pl, dummy_iat)
    mem = torch.cuda.max_memory_allocated(device_gpu) / (1024 ** 2)
    print(f"🔹 GPU Peak Memory: {mem:.2f} MB")

    # ======= 指标三：单流推理延迟 (Inference Latency) =======
    # 预热 GPU
    for _ in range(100):
        _ = model(dummy_imgs, dummy_pl, dummy_iat)
    
    torch.cuda.synchronize()
    start_time = time.time()
    iters = 1000
    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy_imgs, dummy_pl, dummy_iat)
            torch.cuda.synchronize()
    latency = (time.time() - start_time) / iters * 1000
    print(f"🔹 GPU Latency: {latency:.4f} ms/flow")

    # ======= 指标四：CPU 版本吞吐量 (CPU Throughput) =======
    model.to(device_cpu)
    cpu_imgs = dummy_imgs.to(device_cpu)
    cpu_pl = dummy_pl.to(device_cpu)
    cpu_iat = dummy_iat.to(device_cpu)
    
    # 模拟大 Batch 以测试极限吞吐
    test_batch = 128
    batch_imgs = cpu_imgs.repeat(test_batch, 1, 1, 1)
    batch_pl = cpu_pl.repeat(test_batch, 1)
    batch_iat = cpu_iat.repeat(test_batch, 1)
    
    start_time = time.time()
    iters_cpu = 50
    with torch.no_grad():
        for _ in range(iters_cpu):
            _ = model(batch_imgs, batch_pl, batch_iat)
    
    total_flows = test_batch * iters_cpu
    throughput = total_flows / (time.time() - start_time)
    print(f"🔹 CPU Throughput: {throughput:.2f} flows/sec")
    print("="*30)

if __name__ == "__main__":
    benchmark()