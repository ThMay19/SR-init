import sys
import heapq
import copy
from xautodl.models import get_cell_based_tiny_net
import numpy as np
from model.VGG_cifar import *
import torch
from torchvision.models import *
from model.samll_resnet import *

def get_model(args):
    # Note that this get_model function only for getting random initialized model
    print(f"=> Getting {args.arch}")
    if args.arch == 'ResNet18':
        model = resnet18(pretrained=args.pretrained)
    elif args.arch == 'ResNet34':
        model = resnet34(pretrained=args.pretrained)
    elif args.arch == 'ResNet50':
        model = resnet50(pretrained=args.pretrained)
    elif args.arch == 'ResNet101':
        model = resnet101(pretrained=args.pretrained)
    elif args.arch == 'cvgg11_bn':
        model = cvgg11_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('./pretrained_model/cvgg11_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('./pretrained_model/cvgg11_bn/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet20':
        model = resnet20(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('./pretrained_model/resnet20.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            elif args.set == 'cifar100':
                ckpt = torch.load('./pretrained_model/resnet20/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet32':
        model = resnet32(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('./pretrained_model/resnet32.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            elif args.set == 'cifar100':
                ckpt = torch.load('./pretrained_model/resnet32/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet44':
        model = resnet44(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('./pretrained_model/resnet44.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            elif args.set == 'cifar100':
                ckpt = torch.load('./pretrained_model/resnet44/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet56':
        model = resnet56(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('./pretrained_model/resnet56.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            elif args.set == 'cifar100':
                ckpt = torch.load('./pretrained_model/resnet56/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet110':
        model = resnet110(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('./pretrained_model/resnet110.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            elif args.set == 'cifar100':
                ckpt = torch.load('./pretrained_model/resnet110/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)

    else:
        assert "the model has not prepared"
    # if the model is loaded from torchvision, then the codes below do not need.
    if args.set in ['cifar10', 'cifar100']:
        if args.pretrained:
            model.load_state_dict(ckpt)
        else:
            print('No pretrained model')
    else:
        print('Not mentioned dataset')
    return model


def get_NAS_model(args, api):
    if args.arch[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        config = api.get_net_config(int(args.arch), args.set)
        model = get_cell_based_tiny_net(config)
        if args.pretrained:
            ckpt = torch.load('/public/ly/ASE/pretrained_model/tss_{}/cifar10/score.pt'.format(str(args.arch)), map_location='cuda:%d' % args.gpu)
            model.load_state_dict(ckpt)
    else:
        sys.exit(0)

    return model

