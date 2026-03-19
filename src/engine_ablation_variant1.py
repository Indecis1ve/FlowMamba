import torch, math, sys
import util.misc as misc, util.lr_sched as lr_sched
from timm.utils import accuracy
from util.misc import NativeScalerWithGradNormCount as NativeScaler

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, amp_autocast, max_norm=0, mixup_fn=None, log_writer=None, args=None):
    model.train(True); metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    accum_iter = args.accum_iter; optimizer.zero_grad()
    
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, 20, f'Epoch: [{epoch}]')):
        if data_iter_step % accum_iter == 0: lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        # [Nety 解包] 提取多模态特征
        imgs, pl, iat, targets = batch
        imgs = imgs.to(device, non_blocking=True)
        pl = pl.to(device, non_blocking=True)
        iat = iat.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with amp_autocast():
            outputs = model(imgs, pl=pl, iat=iat)
            loss = criterion(outputs, targets)
            
        loss_value = loss.item()
        if not math.isfinite(loss_value): sys.exit(1)
        
        if isinstance(loss_scaler, NativeScaler):
            loss /= accum_iter
            loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0: optimizer.zero_grad()
        else:
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss(); model.eval(); metric_logger = misc.MetricLogger(delimiter="  ")
    for batch in metric_logger.log_every(data_loader, 10, 'Test:'):
        imgs, pl, iat, target = batch
        imgs = imgs.to(device, non_blocking=True)
        pl = pl.to(device, non_blocking=True)
        iat = iat.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            output = model(imgs, pl=pl, iat=iat)
            loss = criterion(output, target)
            
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item() / 100, n=imgs.shape[0])
        metric_logger.meters['acc5'].update(acc5.item() / 100, n=imgs.shape[0])
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
