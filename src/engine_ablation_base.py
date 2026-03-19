import torch, math, util.misc as misc, util.lr_sched as lr_sched
from timm.utils import accuracy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, amp_autocast, max_norm=0, mixup_fn=None, log_writer=None, args=None):
    model.train(True); metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    for data_iter_step, (imgs, targets) in enumerate(metric_logger.log_every(data_loader, 20, f'Epoch: [{epoch}]')):
        if data_iter_step % args.accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with amp_autocast():
            outputs = model(imgs) # [Nety 消融] 仅传 imgs
            loss = criterion(outputs, targets)
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), update_grad=(data_iter_step + 1) % args.accum_iter == 0)
        if (data_iter_step + 1) % args.accum_iter == 0: optimizer.zero_grad()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device):
    model.eval(); metric_logger = misc.MetricLogger(delimiter="  ")
    pred_all, target_all = [], []
    for imgs, target in metric_logger.log_every(data_loader, 10, 'Test:'):
        imgs, target = imgs.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            output = model(imgs) # [Nety 消融] 仅传 imgs
        acc1, _ = accuracy(output, target, topk=(1, 5))
        metric_logger.meters['acc1'].update(acc1.item() / 100, n=imgs.shape[0])
        pred_all.extend(torch.argmax(output, dim=1).cpu()); target_all.extend(target.cpu())
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
