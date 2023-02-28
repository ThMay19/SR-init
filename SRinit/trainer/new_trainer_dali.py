import math
import time
import torch
import tqdm
from torch.cuda.amp import autocast, GradScaler

from args import args
from utils.logging import AverageMeter, ProgressMeter
from utils.utils import accuracy


__all__ = ["train_ImageNet", "validate_ImageNet"]

def adjust_learning_rate(optimizer, epoch, step, len_iter):

    if args.lr_type == 'step':
        factor = epoch // 30
        if epoch >= 80:
                factor = factor + 1
        lr = args.lr * (0.1 ** factor)

    elif args.lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch - 5) / (args.epochs - 5)))

    elif args.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.lr * (decay ** (epoch // step))

    elif args.lr_type == 'fixed':
        lr = args.lr
    else:
        raise NotImplementedError

    #Warmup
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_iter) / (5. * len_iter)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # if step == 0:
    #  logger.info('learning_rate: ' + str(lr))


def train_ImageNet(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()

    num_iter = train_loader._size // args.batch_size

    scaler = GradScaler()
    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    train_loader.reset()
    for i, data in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        # measure data loading time
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        if args.gpu is not None:
            images = data[0]["data"].cuda(non_blocking=True)

        target = data[0]["label"].squeeze().long().cuda(non_blocking=True)
        # new
        adjust_learning_rate(optimizer, epoch, i, num_iter)
        with autocast():
            output = model(images)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)

    return top1.avg, top5.avg


def validate_ImageNet(val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        end = time.time()
        for i, data in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = data[0]["data"].cuda(non_blocking=True)

            target = data[0]["label"].squeeze().long().cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

    return top1.avg, top5.avg

