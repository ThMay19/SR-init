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
    # torch.load("./result/baseline/best_acc_resnet110_on_cifar100.pt")#71.65
    # torch.load("./result/random_acc/random_replaced_acc_{}_seed{}.pt".format(args.set, args.arch, args.random_seed))
    # replaced_acc = [x / 100 for x in replaced_acc]
    best_acc = 0.7165
    replaced_acc = [0.0643, 0.11460000000000001, 0.17300000000000001, 0.3447, 0.0553, 0.044199999999999996, 0.0379,
                    0.24780000000000002, 0.1063, 0.0189, 0.0352, 0.0189, 0.192, 0.0311, 0.0167, 0.016399999999999998,
                    0.0196, 0.0317,
                    0.0139, 0.0451, 0.0688, 0.1225, 0.0242, 0.17550000000000002, 0.057699999999999994, 0.0425, 0.0189,
                    0.18289999999999998, 0.0361, 0.0308, 0.06709999999999999, 0.0843, 0.0582, 0.0692,
                    0.018500000000000003, 0.0342,
                    0.0282, 0.19920000000000002, 0.034300000000000004, 0.1659, 0.045700000000000005,
                    0.054400000000000004, 0.1275,
                    0.2781, 0.0575, 0.11539999999999999, 0.1968, 0.3544, 0.33590000000000003, 0.4681,
                    0.48619999999999997, 0.5608,
                    0.5667, 0.5781000000000001]
    print(replaced_acc, best_acc)

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
    delta_r56_c10 = [0.6135, 0.516, 0.6361, 0.7981, 0.7502, 0.6853, 0.8024,
                     0.7281, 0.5719, 0.8304, 0.7592, 0.832, 0.7277, 0.7185, 0.6498,
                     0.7615, 0.8095, 0.4507, 0.6273, 0.3921, 0.6664, 0.5793, 0.1467, 0.055, 0.0118, 0.0085, 0.0043]
    delta_r56_c10_best = 0.9305
    delta_r56_c100 = [0.689, 0.5453, 0.706, 0.4635, 0.6334, 0.6968, 0.6915,
                      0.7067, 0.6655, 0.7095, 0.6748, 0.7025, 0.6958, 0.6188, 0.6594,
                      0.6641, 0.6874, 0.7011, 0.6997, 0.6504, 0.6727, 0.6255, 0.6182, 0.4251, 0.3663, 0.1493, 0.0767]
    delta_r56_c100_best = 0.7181
    delta_r50_I = [0.7552, 0.5531, 0.5774, 0.7559, 0.5974, 0.5538, 0.7444, 0.7556,
                   0.4897, 0.5141, 0.5499, 0.5561, 0.5778, 0.7558, 0.433, 0.2625]
    delta_r50_I_best = 0.7569

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
    plt.savefig('./result/plt/pruning.pdf')
    return


if __name__ == '__main__':
    # load_acc()
    # plot_estimation_accuracy()
    plot_layer_pruning()
# python Draw_plt.py --arch resnet56 --set cifar10 --num_classes 10 --random_seed 1