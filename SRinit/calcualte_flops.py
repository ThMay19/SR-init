import torch
import argparse
from thop import profile
import torchvision
from trainer.amp_trainer_dali import validate_ImageNet
from trainer.trainer import validate
from utils.Get_dataset import get_dataset

from utils.claculate_latency import compute_latency_ms_pytorch
from utils.get_new_model import get_model

parser = argparse.ArgumentParser(description='Calculating flops and params')

parser.add_argument(
    '--input_image_size',
    type=int,
    default=224,
    help='The input_image_size')
parser.add_argument("--gpu", default=None, type=int, help="Which GPU to use for training")
parser.add_argument("--arch", default=None, type=str, help="arch")
parser.add_argument("--pretrained", action="store_true", help="use pre-trained model")
parser.add_argument("--num_classes", default=10, type=int, help="number of class")
parser.add_argument("--finetune", default=None, action="store_true", help="finetune pre-trained model")
parser.add_argument("--finetuned", action="store_true", help="use finetuned model")
parser.add_argument("--set", help="name of dataset", type=str, default='cifar10')
parser.add_argument("--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
parser.add_argument("--print-freq", default=100, type=int, metavar="N", help="print frequency (default: 10)")
parser.add_argument("--random_seed", default=1, type=int, help="random seed")
args = parser.parse_args()
torch.cuda.set_device(args.gpu)
model = get_model(args).cuda()
model.eval()
criterion = torch.nn.CrossEntropyLoss().cuda()
data = get_dataset(args)

if args.evaluate:
    if args.set in ['cifar10', 'cifar100']:
        acc1, acc5 = validate(data.val_loader, model, criterion, args)
        print(acc1)
    else:
        acc1, acc5 = validate_ImageNet(data.val_loader, model, criterion, args)

    print('Acc is {}'.format(acc1))

# calculate model size
input_image_size = 32
print('image size is {}'.format(input_image_size))
input_image = torch.randn(1, 3, input_image_size, input_image_size).cuda()
flops, params = profile(model, inputs=(input_image,))
latency = compute_latency_ms_pytorch(model, input_image, iterations=None)

print('Params: %.2f' % (params))
print('Flops: %.2f' % (flops))
print('Latency: %.2f' % (latency))

'''
python calcualte_flops.py --gpu 0 --arch resnet56 --pretrained --evaluate --set cifar10 --num_classes 10


python calcualte_flops.py --gpu 0 --arch resnet56 --pretrained --evaluate --set cifar100 --num_classes 100


python calcualte_flops.py --gpu 0 --arch resnet56_cut10 --finetuned --evaluate --set cifar10 --num_classes 10


python calcualte_flops.py --gpu 0 --arch resnet56_cut7 --finetuned --evaluate --set cifar100 --num_classes 100


python calcualte_flops.py --gpu 0 --arch ResNet50 --pretrained --evaluate --set imagenet_dali --num_classes 1000


python calcualte_flops.py --gpu 0 --arch ResNet50_cut3 --finetuned --evaluate --set imagenet_dali --num_classes 1000


'''