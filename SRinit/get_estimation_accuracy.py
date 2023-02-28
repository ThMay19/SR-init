import argparse
from torchvision import models

from utils.Get_model import get_model
from model.samll_resnet import resnet56, resnet110
from trainer.amp_trainer_dali import validate_ImageNet
from trainer.trainer import validate
from utils.Get_dataset import get_dataset
from utils.get_params import get_layer_params
from utils.utils import Logger, set_random_seed, set_gpu, get_logger
import torch

layer_num = {'resnet56': 28, 'resnet44': 22, 'resnet32': 18, 'resnet20': 10, 'cvgg16_bn': 13, 'resnet110': 55}
layer_list = {
    'cvgg16_bn': [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40, 1, 4, 6],
    'ResNet50': [3, 4, 6, 3],
    'ResNet101': [3, 4, 23, 3]
}

parser = argparse.ArgumentParser(description='Calculating flops and params')
parser.add_argument("--gpu", default='0', type=int, help="Which GPU to use for training")
parser.add_argument("--arch", default='resnet56', type=str, help="arch")
parser.add_argument("--num_classes", default=100, type=int, help="number of class")
parser.add_argument("--set", help="name of dataset", type=str, default='cifar100')
parser.add_argument("--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
parser.add_argument("--print-freq", default=100, type=int, metavar="N", help="print frequency (default: 10)")
parser.add_argument("--multigpu", default=None, type=lambda x: [int(a) for a in x.split(",")],
                    help="Which GPUs to use for multigpu training")
parser.add_argument("--random_seed", default=1, type=int, help="random seed")
parser.add_argument("--pretrained", default=False, help="use pre-trained model")

args = parser.parse_args()
torch.cuda.set_device(args.gpu)
criterion = torch.nn.CrossEntropyLoss().cuda()
data = get_dataset(args)


def replace_layer_initialization(path):
    """ Run the methods on the data and then saves it to out_path. """
    if args.arch == 'resnet56':
        model1 = resnet56(num_classes=args.num_classes)
        model2 = resnet56(num_classes=args.num_classes)
        if args.set == 'cifar10':
            ckpt = torch.load('./pretrained_model/train/resnet56/cifar10/scores.pt',
                              map_location='cuda:%d' % args.gpu)
        elif args.set == 'cifar100':
            ckpt = torch.load("./pretrained_model/train/resnet56/cifar100/scores.pt",
                              map_location='cuda:%d' % args.gpu)
    if args.arch == 'resnet110':
        model1 = resnet110(num_classes=args.num_classes)
        model2 = resnet110(num_classes=args.num_classes)
        if args.set == 'cifar10':
            ckpt = torch.load("./pretrained_model/resnet/resnet110/cifar10/resnet110.th",
                              map_location='cuda:%d' % args.gpu)
        elif args.set == 'cifar100':
            ckpt = torch.load("./pretrained_model/train/resnet110/cifar100/scores.pt",
                              map_location='cuda:%d' % args.gpu)
    elif args.arch == 'ResNet50':
        model1 = models.resnet50(pretrained=False)
        model2 = models.resnet50(pretrained=False)
        ckpt = torch.load(
            "./pretrained_model/train/ResNet50/resnet50.pth")
    elif args.arch == 'ResNet101':
        model1 = models.resnet101(pretrained=False)
        model2 = models.resnet101(pretrained=False)
        ckpt = torch.load(
            "./pretrained_model/train/ResNet101/resnet101.pth")

    ckpt2 = torch.load(path, map_location='cuda:%d' % args.gpu)
    model2.load_state_dict(ckpt2)

    model1 = set_gpu(args, model1)
    model2 = set_gpu(args, model2)

    # imagenet or cifar10 or cifar100
    model1.eval()
    model2.eval()
    list = []
    orginal_state_list = get_layer_params(model2)

    if args.evaluate:
        model1.load_state_dict(ckpt)
        if args.set in ['cifar10', 'cifar100']:
            acc1, acc5 = validate(data.val_loader, model1, criterion, args)
        else:
            acc1, acc5 = validate_ImageNet(data.val_loader, model1, criterion, args)

        print('The baseline Acc is {}'.format(acc1))

    if args.arch == 'resnet56' or args.arch == 'resnet20' or args.arch == 'resnet110':
        model1.load_state_dict(ckpt)
        acc1, acc5 = validate(data.val_loader, model1, criterion, args)
        torch.save(acc1, "./result/baseline/best_acc_{}_on_{}.pt".format(
            args.arch, args.set))
        x = (layer_num[args.arch] - 1) // 3
        for i in range(1, layer_num[args.arch]):
            if i == 0:
                model1.load_state_dict(ckpt)
                model1.conv1.load_state_dict(orginal_state_list[0])
                model1.bn1.load_state_dict(orginal_state_list[1])
                acc1, acc5 = validate(data.val_loader, model1, criterion, args)
                list.append(acc1)
            elif 0 < i <= x:
                model1.load_state_dict(ckpt)
                model1.layer1[i - 1].load_state_dict(orginal_state_list[i + 1])
                acc1, acc5 = validate(data.val_loader, model1, criterion, args)
                list.append(acc1)
            elif x < i <= 2 * x:
                model1.load_state_dict(ckpt)
                model1.layer2[i - 1 - x].load_state_dict(orginal_state_list[i + 1])
                acc1, acc5 = validate(data.val_loader, model1, criterion, args)
                list.append(acc1)
            elif 2 * x < i <= 3 * x:
                model1.load_state_dict(ckpt)
                model1.layer3[i - 1 - (2 * x)].load_state_dict(orginal_state_list[i + 1])
                acc1, acc5 = validate(data.val_loader, model1, criterion, args)
                list.append(acc1)
    elif args.arch == 'ResNet50' or args.arch == 'ResNet101':
        start = 2
        model1.load_state_dict(ckpt)
        acc1, acc5 = validate(data.val_loader, model1, criterion, args)
        torch.save(acc1, "./finetune/result_test/baseline/best_acc_{}_on_{}.pt".format(
            args.arch, args.set))
        for n, m in enumerate(layer_list[args.arch]):
            if n == 0:
                for x in range(m):
                    model1.load_state_dict(ckpt)
                    model1.layer1[x].load_state_dict(orginal_state_list[x + start])
                    acc1, acc5 = validate_ImageNet(data.val_loader, model1, criterion, args)
                    print(acc1)
                    list.append(acc1)
            elif n == 1:
                start += layer_list[args.arch][0]
                for x in range(m):
                    model1.load_state_dict(ckpt)
                    model1.layer2[x].load_state_dict(orginal_state_list[x + start])
                    acc1, acc5 = validate_ImageNet(data.val_loader, model1, criterion, args)
                    print(acc1)
                    list.append(acc1)
            elif n == 2:
                start += layer_list[args.arch][1]
                for x in range(m):
                    model1.load_state_dict(ckpt)
                    model1.layer3[x].load_state_dict(orginal_state_list[x + start])
                    acc1, acc5 = validate_ImageNet(data.val_loader, model1, criterion, args)
                    print(acc1)
                    list.append(acc1)
            elif n == 3:
                start += layer_list[args.arch][2]
                for x in range(m):
                    model1.load_state_dict(ckpt)
                    model1.layer4[x].load_state_dict(orginal_state_list[x + start])
                    acc1, acc5 = validate_ImageNet(data.val_loader, model1, criterion, args)
                    print(acc1)
                    list.append(acc1)


    elif args.arch == 'cvgg16_bn':
        for i in range(layer_num[args.arch]):
            model1.load_state_dict(ckpt)
            model1.features[layer_list[args.arch][i]].load_state_dict(orginal_state_list[2 * i])
            model1.features[layer_list[args.arch][i] + 1].load_state_dict(orginal_state_list[2 * i + 1])
            acc1, acc5 = validate(data.val_loader, model1, criterion, args)
            list.append(acc1)
    print(list)
    return list


def main():
    if args.random_seed is not None:
        set_random_seed(args.random_seed)
    path = "./pretrained_model/random_model/random_{}_seed{}_{}.th".format(
        args.arch, args.random_seed, args.set)
    layer_acc_list = replace_layer_initialization(path)
    torch.save(layer_acc_list,
               "./result/random_acc/random_replaced_acc_{}_seed{}_on_{}.pt".format(args.arch, args.random_seed, args.set))


def get_random_model():
    if args.random_seed is not None:
        set_random_seed(args.random_seed)
    model = get_model(args)
    torch.save(model.state_dict(),
               "./pretrained_model/random_model/random_{}_seed{}_{}.th".format(args.arch, args.random_seed,args.set))

if __name__ == "__main__":
    # get_random_model()   #get random initialized model first
    main()

'''
python get_estimation_accuracy.py --arch resnet56 --set cifar10 --num_classes 10 --random_seed 1 --evaluate
python get_estimation_accuracy.py --arch ResNet50 --set imagenet_dali --num_classes 1000 --random_seed 1 --evaluate
'''

