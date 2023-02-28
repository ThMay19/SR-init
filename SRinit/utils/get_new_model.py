import sys
from torchvision import models
from model import ResNet_ImageNet
from model import samll_resnet
import torch
from torchvision.models import *
from utils.finetune_layer import get_cresnet_layer_params, load_cresnet_layer_params, load_vgg_layer_params, \
    get_Iresnet_layer_params, load_Iresnet_layer_params

def get_model(args):
    # Note that you can train your own models using train(_for_iamgenet).py
    print(f"=> Getting {args.arch}")
    if args.arch == 'ResNet34':
        model = resnet34(pretrained=False)
        if args.pretrained:
            ckpt = torch.load("./pretrained_model/train/resnet34/resnet34.pth",
                              map_location='cuda:%d' % args.gpu)
            model.load_state_dict(ckpt)
    if args.arch == 'ResNet50':
        model = resnet50(pretrained=False)
        if args.pretrained:
            ckpt = torch.load("./pretrained_model/train/ResNet50/resnet50.pth",
                              map_location='cuda:%d' % args.gpu)
            model.load_state_dict(ckpt)
    elif args.arch == 'resnet56':
        model = samll_resnet.resnet56(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load(
                    "./pretrained_model/train/resnet56/cifar10/scores.pt",
                    map_location='cuda:%d' % args.gpu)
                model.load_state_dict(ckpt)
                # ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            elif args.set == 'cifar100':
                ckpt = torch.load(
                    "./pretrained_model/train/resnet56/cifar100/scores.pt",
                    map_location='cuda:%d' % args.gpu)
                model.load_state_dict(ckpt)

    elif args.arch == 'resnet56_cut10':
        # {1, 8, 17, 19, 21, 22, 23, 24, 25, 26}
        model = samll_resnet.resnet56_cut10(num_classes=args.num_classes)
        if args.finetune:
            orginal_model = samll_resnet.resnet56(num_classes=args.num_classes)
            ckpt = torch.load("./pretrained_model/train/resnet56/cifar10/scores.pt",
                              map_location='cuda:%d' % args.gpu)
            orginal_model.load_state_dict(ckpt)
            orginal_state_list = get_cresnet_layer_params(orginal_model)
            remain_list = [[0, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 2]]
            model = load_cresnet_layer_params(orginal_state_list, model, remain_list, num_of_block=9)
            print('Load pretrained weights from the original model')
        if args.finetuned:
            ckpt = torch.load("./finetuned_model/resnet56_cut10/cifar10/scores.pt",
                              map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet56_cut7':
        # {1, 3, 17, 23, 24, 25, 26}
        model = samll_resnet.resnet56_cut7(num_classes=args.num_classes)
        if args.finetune:
            orginal_model = samll_resnet.resnet56(num_classes=args.num_classes)
            ckpt = torch.load("./pretrained_model/train/resnet56/cifar100/scores.pt",
                              map_location='cuda:%d' % args.gpu)
            orginal_model.load_state_dict(ckpt)
            orginal_state_list = get_cresnet_layer_params(orginal_model)
            remain_list = [[0, 2, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4]]
            model = load_cresnet_layer_params(orginal_state_list, model, remain_list, num_of_block=9)
            print('Load pretrained weights from the original model')
        if args.finetuned:
            ckpt = torch.load(
                "./finetuned_model/resnet56_cut7/cifar100/scores.pt",
                map_location='cuda:%d' % args.gpu)

    elif args.arch == 'ResNet50_cut6':
        # {1, 8, 9, 10, 14, 15}
        model = ResNet_ImageNet.Iresnet50_cut6(pretrained=False)
        if args.finetune:
            orginal_model = models.resnet50(pretrained=False)
            ckpt = torch.load("./pretrained_model/train/ResNet50/resnet50.pth",
                              map_location='cuda:%d' % args.gpu)
            orginal_model.load_state_dict(ckpt)
            orginal_state_list = get_Iresnet_layer_params(orginal_model)
            remain_list = [[0, 2], [0, 1, 2, 3], [0, 4, 5], [0]]  
            model = load_Iresnet_layer_params(orginal_state_list, model, remain_list,
                                              num_of_block=[3, 4, 6, 3])
            print('Load pretrained weights from the original model')
        if args.finetuned:
            ckpt = torch.load("./finetuned_model/ResNet50_cut6/imagenet_dali/1/scores.pt",
                              map_location='cuda:%d' % args.gpu)


    elif args.arch == 'ResNet50_cut3':
        model = ResNet_ImageNet.Iresnet50_cut3(pretrained=False)
        if args.finetune:
            orginal_model = models.resnet50(pretrained=False)
            ckpt = torch.load("./pretrained_model/train/ResNet50/resnet50.pth",
                              map_location='cuda:%d' % args.gpu)
            orginal_model.load_state_dict(ckpt)
            orginal_state_list = get_Iresnet_layer_params(orginal_model)

            remain_list = [[0, 1, 2], [0, 1, 2, 3], [0, 2, 3, 4, 5], [0]]
            model = load_Iresnet_layer_params(orginal_state_list, model, remain_list,
                                              num_of_block=[3, 4, 6, 3])
            print('Load pretrained weights from the original model')
        if args.finetuned:
            ckpt = torch.load(
                "./finetuned_model/ResNet50_cut3/imagenet_dali/1/scores.pt",
                map_location='cuda:%d' % args.gpu)
    else:
        assert "the model has not prepared"
    # if the model is loaded from pretrained_model, then the codes below do not need.
    if args.set in ['cifar10', 'cifar100', 'imagenet_dali']:
        if args.finetuned:
            model.load_state_dict(ckpt)
        # else:
        #     print('No pretrained model')
    else:
        print('Not mentioned dataset')
    return model
