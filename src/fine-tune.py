import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import count_parameters

import models_net_mamba
from contextlib import suppress
from engine import train_one_epoch, evaluate
import torch.nn.functional as F

# [Nety 核心修改 1]：引入路线 A 制作的多模态 Dataset
from dataset.dataset_common import MultimodalTrafficDataset

# ==================== [Nety 路径修复] ====================
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# =======================================================

def get_args_parser():
    parser = argparse.ArgumentParser('NetMamba fine-tuning for traffic classification', add_help=False)

    # [Nety 新增] 支持未来数据增强，但多模态下强制禁用
    parser.add_argument('--mixup', type=float, default=0.0, help='mixup alpha (default: 0)')
    parser.add_argument('--cutmix', type=float, default=0.0, help='cutmix alpha (default: 0)')

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--save_steps_freq', default=5000, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations...')
    # Model parameters
    parser.add_argument('--model', default='net_mamba_classifier', type=str, metavar='MODEL')
    parser.add_argument('--input_size', default=40, type=int)
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT')
    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=1e-3)
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    # Augmentation parameters
    parser.add_argument('--smoothing', type=float, default=0.1)
    # Dataset parameters
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--nb_classes', default=20, type=int)
    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--log_dir', default='./output_dir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dist_eval', action='store_true', default=False)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    # distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    return parser


def build_dataset(data_split, args):
    dataset = MultimodalTrafficDataset(data_path=args.data_path, split=data_split)
    return dataset


def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train = build_dataset('train', args)
    dataset_val = build_dataset('valid', args)
    dataset_test = build_dataset('test', args)

    # ==================== [Nety 修复] ====================
    if args.distributed:  # ← 改成这个！不再是 if True
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank,
                                                            shuffle=True)
        if args.dist_eval:
            sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank,
                                                              shuffle=True)
            sampler_test = torch.utils.data.DistributedSampler(dataset_test, num_replicas=num_tasks, rank=global_rank,
                                                               shuffle=True)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    # ====================================================

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size,
                                                    num_workers=args.num_workers, pin_memory=args.pin_mem,
                                                    drop_last=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, batch_size=args.batch_size,
                                                  num_workers=args.num_workers, pin_memory=args.pin_mem,
                                                  drop_last=False)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, sampler=sampler_test, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, pin_memory=args.pin_mem,
                                                   drop_last=False)

    # [Nety 硬防护] 多模态严禁 Mixup/Cutmix
    if args.mixup > 0 or args.cutmix > 0:
        print("[Nety WARNING] Mixup/Cutmix 在 Hybrid Multimodal NetMamba 中已被强制禁用！（PL/IAT 无法线性插值）")
    mixup_fn = None

    model = models_net_mamba.__dict__[args.model](num_classes=args.nb_classes, drop_path_rate=args.drop_path)
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()

    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"[Nety 监测台] 引擎点火！开始 {args.epochs} 个 Epoch 的多模态训练...")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # 1. 前向传播与反向更新
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            torch.cuda.amp.autocast, args.clip_grad, mixup_fn,
            log_writer=log_writer, args=args
        )

        # 2. 定期保存 Checkpoint 权重
        if args.output_dir:
            if (epoch % 10 == 0 or epoch + 1 == args.epochs):
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

        # 3. 在验证集上评估当前模型
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} valid images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        # 4. 写入 TensorBoard 日志
        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            # ==================== [Nety 终极日志降维防御] ====================
            clean_log_stats = {}
            for k, v in log_stats.items():
                if isinstance(v, (torch.Tensor, np.ndarray)):
                    # 如果是张量或数组（无论多维还是单维），一律转为原生 List 或标量
                    clean_log_stats[k] = v.tolist()
                elif hasattr(v, 'item'):
                    # 如果是 Numpy 的独立标量
                    try:
                        clean_log_stats[k] = v.item()
                    except ValueError:
                        clean_log_stats[k] = str(v)
                else:
                    clean_log_stats[k] = v

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(clean_log_stats) + "\n")
            # =============================================================
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('[Nety 报告] 训练圆满结束！总耗时: {}'.format(total_time_str))
    # ===============================================================


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)