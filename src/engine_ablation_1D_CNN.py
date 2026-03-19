import torch, math, sys
import util.misc as misc, util.lr_sched as lr_sched
from timm.utils import accuracy
from util.misc import NativeScalerWithGradNormCount as NativeScaler

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, amp_autocast, max_norm=0, mixup_fn=None, log_writer=None, args=None):
    model.train(True); metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    accum_iter = args.accum_iter; optimizer.zero_grad()
    for data_iter_step, (imgs, targets) in enumerate(metric_logger.log_every(data_loader, 20, f'Epoch: [{epoch}]')):
        if data_iter_step % accum_iter == 0: lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with amp_autocast():
            outputs = model(imgs)
            loss = criterion(outputs, targets)
        loss_value = loss.item()
        if not math.isfinite(loss_value): print(f"Loss is {loss_value}, stopping"); sys.exit(1)
        if isinstance(loss_scaler, NativeScaler):
            loss /= accum_iter
            loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0: optimizer.zero_grad()
        else:
            loss.backward(); optimizer.step(); optimizer.zero_grad()
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss(); model.eval(); metric_logger = misc.MetricLogger(delimiter="  ")
    for imgs, target in metric_logger.log_every(data_loader, 10, 'Test:'):
        imgs, target = imgs.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            output = model(imgs)
            loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = imgs.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item() / 100, n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item() / 100, n=batch_size)
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
