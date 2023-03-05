import os
import sys
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
from args import args
import datetime
from trainer.new_trainer_dali import train_ImageNet, validate_ImageNet
from trainer.trainer import validate, train
from utils.Get_dataset import get_dataset
from utils.get_new_model import get_model
from utils.utils import set_random_seed, set_gpu, Logger, get_logger, get_lr
from utils.warmup_lr import cosine_lr
import utils.common as utils


def main():
    print(args)
    sys.stdout = Logger('print process.log', sys.stdout)

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    main_worker(args)


def main_worker(args):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('finetuned_model/' + args.arch + '/' + args.set + '/' + args.location):
        os.makedirs('finetuned_model/' + args.arch + '/' + args.set + '/' + args.location, exist_ok=True)
    logger = get_logger('finetuned_model/' + args.arch + '/' + args.set + '/' + args.location + '/logger' + now + '.log')
    logger.info(args.arch)
    logger.info(args.set)
    logger.info(args.batch_size)
    logger.info(args.weight_decay)
    logger.info(args.lr)
    logger.info(args.epochs)
    logger.info(args.lr_decay_step)
    logger.info(args.num_classes)
    logger.info(args.random_seed)
    logger.info(args.lr_type)
    logger.info(args.momentum)
    logger.info(args.location)
    # logger.info(args)

    model = get_model(args)
    model = set_gpu(args, model)
    logger.info(model)
    criterion = nn.CrossEntropyLoss().cuda()
    # label_smooth
    criterion_smooth = utils.CrossEntropyLabelSmooth(args.num_classes, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    data = get_dataset(args)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    # create recorder
    args.start_epoch = args.start_epoch or 0

    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        # scheduler(epoch, iteration=None)
        # cur_lr = get_lr(optimizer)
        # logger.info(f"==> CurrentLearningRate: {cur_lr}")
        # scheduler.step()
        if args.set == 'imagenet_dali':
            # for imagenet
            #label_smooth
            train_acc1, train_acc5 = train_ImageNet(data.train_loader, model, criterion_smooth, optimizer, epoch, args)

            #train_acc1, train_acc5 = train_ImageNet(data.train_loader, model, criterion, optimizer, epoch, args)
            acc1, acc5 = validate_ImageNet(data.val_loader, model, criterion, args)
        else:
            #  for small datasets
            train_acc1, train_acc5 = train(data.train_loader, model, criterion, optimizer, epoch, args)
            acc1, acc5 = validate(data.val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)
        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                torch.save(model.state_dict(), 'finetuned_model/' + args.arch + '/' + args.set + '/' + args.location + "/scores.pt")
                logger.info(best_acc1)

if __name__ == "__main__":
    main()

