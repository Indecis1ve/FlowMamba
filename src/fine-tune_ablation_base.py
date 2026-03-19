import os
import sys

# ==================== [Nety 核心修复 1：路径优先级置顶] ====================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# =====================================================================

import argparse
import datetime
import json
import numpy as np
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from timm.models.layers import trunc_normal_
import util.lr_decay as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# 导入 Baseline 专用组件
import models_net_mamba_ablation_base as models_net_mamba
from engine_ablation_base import train_one_epoch, evaluate
from dataset.dataset_common_ablation_base import MultimodalTrafficDataset

def get_args_parser():
    parser = argparse.ArgumentParser('NetMamba fine-tuning for Baseline Ablation', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--model', default='net_mamba_classifier', type=str)
    parser.add_argument('--input_size', default=40, type=int)
    parser.add_argument('--drop_path', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=1e-3)
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--output_dir', default='./output/ablation_base_run1')
    parser.add_argument('--log_dir', default='./output/ablation_base_run1')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--nb_classes', default=20, type=int)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--dist_backend', default='nccl', type=str)
    return parser

def main(args):
    # ==================== [Nety 核心修复 2：全套单机环境补丁] ====================
    # 模拟 torchrun/slurm 的环境变量，彻底解决 misc.py 的 KeyError
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29506'
    # ===========================================================================

    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train = MultimodalTrafficDataset(split='train', seed=args.seed)
    dataset_val = MultimodalTrafficDataset(split='valid', seed=args.seed)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    # 实例化 Baseline 模型
    model = models_net_mamba.__dict__[args.model](num_classes=args.nb_classes, drop_path_rate=args.drop_path)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[Nety 报告] Baseline 模型参数量: {n_parameters/1e6:.2f}M')

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()
    criterion = torch.nn.CrossEntropyLoss()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"[Nety 报告] 启动 Baseline 400 Epoch 训练...")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, torch.cuda.amp.autocast, args=args)
        
        if args.output_dir and (epoch % 50 == 0 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Epoch {epoch} Accuracy: {test_stats['acc1']:.2%}")
        max_accuracy = max(max_accuracy, test_stats['acc1'])

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            clean_log_stats = {k: (v.tolist() if isinstance(v, (torch.Tensor, np.ndarray)) else (v.item() if hasattr(v, 'item') else v)) for k, v in log_stats.items()}
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(clean_log_stats) + "\n")

    print(f'[Nety 报告] 训练完成！')

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
