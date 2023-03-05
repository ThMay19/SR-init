from matplotlib import pyplot as plt
from args import args
import torch

layer_num = {'resnet20': 10, 'resnet56': 28, 'resnet44': 22, 'resnet32': 18, 'resnet110': 54, 'ResNet50': 16}


def load_acc():
    best_acc = torch.load("./result/baseline/best_acc_{}_on_{}.pt".format(args.arch, args.set))
    print("Best Accuracy:", best_acc)
    replaced_acc = torch.load(
        "./result/random_acc/random_replaced_acc_{}_seed{}_on_{}.pt".format(args.arch, args.random_seed, args.set))
    delta_acc = {'{}'.format(i): round((best_acc - replaced_acc[i]) / 100, 4) for i in range(layer_num[args.arch])}
    print("Accuracy Drop:", delta_acc)


def plot_estimation_accuracy():
    best_acc = torch.load("./result/baseline/best_acc_resnet110_on_cifar100.pt") / 100  
    replaced_acc = torch.load(
        "./result/random_acc/random_replaced_acc_resnet110_seed{}_on_cifar100.pt".format(args.random_seed))
    replaced_acc = [(x / 100) for x in replaced_acc]

    fig = plt.figure(figsize=(6, 3))
    plt.rc('font', family='Times New Roman')
    total_list = [replaced_acc]
    best_list = [best_acc]
    remove_layer = [3]
    remain_layer = [51]
    name = ['ResNet110 on CIFAR-100']
    import heapq
    from matplotlib.ticker import MaxNLocator

    min_number = heapq.nsmallest(remove_layer[0], total_list[0])
    min_index = map(total_list[0].index, heapq.nsmallest(remove_layer[0], total_list[0]))
    max_number = heapq.nlargest(remain_layer[0], total_list[0])
    max_index = map(total_list[0].index, heapq.nlargest(remain_layer[0], total_list[0]))

    min_index = map(lambda x: x + 1, min_index)
    max_index = map(lambda x: x + 1, max_index)

    plt.bar(list(min_index), min_number, color='steelblue')
    plt.bar(list(max_index), max_number, color='steelblue')
    plt.ylim(0, 1)
    plt.yticks([0.2, 0.4, 0.6, 0.8])
    plt.xticks([6, 12, 18, 24, 30, 36, 42, 48, 54])
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.axhline(y=best_list[0], linestyle='--')
    plt.legend(['origin', 'estimation'], loc=2)
    plt.title(name[0])
    plt.xlabel('Layer')
    plt.tight_layout()
    plt.savefig('./result/plt/estimation_accuracy.pdf')
    return


def plot_layer_pruning():
    delta_r56_c10_best = torch.load("./result/baseline/best_acc_resnet56_on_cifar10.pt") / 100
    delta_r56_c10 = torch.load("./result/random_acc/random_replaced_acc_resnet56_seed{}_on_cifar10.pt".format(args.random_seed))
    delta_r56_c10 = [delta_r56_c10_best - (x / 100) for x in delta_r56_c10]

    delta_r56_c100_best = torch.load("./result/baseline/best_acc_resnet56_on_cifar100.pt") / 100
    delta_r56_c100 = torch.load("./result/random_acc/random_replaced_acc_resnet56_seed{}_on_cifar100.pt".format(args.random_seed))
    delta_r56_c100 = [delta_r56_c100_best - (x / 100) for x in delta_r56_c100]

    delta_r50_I_best = torch.load("./result/baseline/best_acc_ResNet50_on_imagenet_dali.pt") / 100
    delta_r50_I = torch.load(
        "./result/random_acc/random_replaced_acc_ResNet50_seed{}_on_imagenet_dali.pt".format(args.random_seed))
    delta_r50_I = [delta_r50_I_best - (x / 100) for x in delta_r50_I]

    fig = plt.figure(figsize=(15, 5), dpi=800)
    plt.rc('font', family='Times New Roman')
    total_list = [delta_r56_c10, delta_r56_c100, delta_r50_I]
    best_list = [delta_r56_c10_best, delta_r56_c100_best, delta_r50_I_best]
    threshold = [0.60, 0.6185, 0.50]
    total_layer = [27, 27, 16]
    remove_layer = []
    remain_layer = []
    for i in range(3):
        temp = sum(ii <= threshold[i] for ii in total_list[i])
        remove_layer.append(temp)
        remain_layer.append(total_layer[i] - temp)

    name = ['ResNet56 on CIFAR-10', 'ResNet56 on CIFAR-100', 'ResNet50 on ImageNet']
    import heapq
    from matplotlib.ticker import MaxNLocator

    for i in range(3):
        min_number = heapq.nsmallest(remove_layer[i], total_list[i])
        min_index = map(total_list[i].index, heapq.nsmallest(remove_layer[i], total_list[i]))
        max_number = heapq.nlargest(remain_layer[i], total_list[i])
        max_index = map(total_list[i].index, heapq.nlargest(remain_layer[i], total_list[i]))
        min_index = map(lambda x: x + 1, min_index)
        max_index = map(lambda x: x + 1, max_index)

        plt.subplot(1, 3, i + 1)
        plt.bar(list(min_index), min_number, color='lightblue')
        plt.bar(list(max_index), max_number, color='steelblue')
        plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
        plt.tick_params(labelsize=15)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.axhline(y=best_list[i], linestyle='--')
        plt.axhline(y=threshold[i], linestyle='--', color='orange')
        plt.title(name[i], fontdict={'size': 22})

    plt.tight_layout()
    plt.rcParams['savefig.dpi'] = 800
    plt.rcParams['figure.dpi'] = 800
    plt.savefig('./result/plt/final_pruning.pdf')
    return


if __name__ == '__main__':
    # load_acc()
    # plot_estimation_accuracy()
    plot_layer_pruning()
# python get_SRinit_plt.py --random_seed 1
