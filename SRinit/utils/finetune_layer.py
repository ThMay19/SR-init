from torchvision import models
import torch

def get_cresnet_layer_params(orginal_model):
    orginal_state_list = []
    orginal_state_list.append(orginal_model.conv1.state_dict())
    orginal_state_list.append(orginal_model.bn1.state_dict())
    for child in orginal_model.layer1.children():
        orginal_state_list.append(child.state_dict())

    for child in orginal_model.layer2.children():
        orginal_state_list.append(child.state_dict())

    for child in orginal_model.layer3.children():
        orginal_state_list.append(child.state_dict())

    orginal_state_list.append(orginal_model.linear.state_dict())
    return orginal_state_list

def get_Iresnet_layer_params(orginal_model):
    orginal_state_list = []
    orginal_state_list.append(orginal_model.conv1.state_dict())
    orginal_state_list.append(orginal_model.bn1.state_dict())
    for child in orginal_model.layer1.children():
        orginal_state_list.append(child.state_dict())

    for child in orginal_model.layer2.children():
        orginal_state_list.append(child.state_dict())

    for child in orginal_model.layer3.children():
        orginal_state_list.append(child.state_dict())

    for child in orginal_model.layer4.children():
        orginal_state_list.append(child.state_dict())

    orginal_state_list.append(orginal_model.fc.state_dict())
    return orginal_state_list


def load_cresnet_layer_params(orginal_state_list, pruned_model, remain_layer_list, num_of_block):
    i = 0
    j = 0
    k = 0
    pruned_model.conv1.load_state_dict(orginal_state_list[0])
    pruned_model.bn1.load_state_dict(orginal_state_list[1])
    for child in pruned_model.layer1.children():
        child.load_state_dict(orginal_state_list[remain_layer_list[0][i] + 2])
        i += 1


    for child in pruned_model.layer2.children():
        if j == 0:
            if remain_layer_list[1][j] != 0:
                print('The first layer in layer 2 is initialized by random')
            else:
                child.load_state_dict(orginal_state_list[remain_layer_list[1][j] + num_of_block + 2])
        else:
            child.load_state_dict(orginal_state_list[remain_layer_list[1][j] + num_of_block + 2])
        j += 1

    for child in pruned_model.layer3.children():
        if k == 0:
            if remain_layer_list[2][k] != 0:
                print('The first layer in layer 3 is initialized by random')
            else:
                child.load_state_dict(orginal_state_list[remain_layer_list[2][k] + 2 * num_of_block + 2])
        else:
            child.load_state_dict(orginal_state_list[remain_layer_list[2][k] + 2 * num_of_block + 2])
        k += 1

    pruned_model.linear.load_state_dict(orginal_state_list[-1])
    return pruned_model

def load_Iresnet_layer_params(orginal_state_list, pruned_model, remain_layer_list, num_of_block):
    i = 0
    j = 0
    k = 0
    l = 0
    pruned_model.conv1.load_state_dict(orginal_state_list[0])
    pruned_model.bn1.load_state_dict(orginal_state_list[1])
    for child in pruned_model.layer1.children():
        if i == 0:
            if remain_layer_list[1][i] != 0:
                print('The first layer in layer 1 is initialized by random')
        child.load_state_dict(orginal_state_list[remain_layer_list[0][i] + 2])
        i += 1


    for child in pruned_model.layer2.children():
        if j == 0:
            if remain_layer_list[1][j] != 0:
                print('The first layer in layer 2 is initialized by random')
            else:
                child.load_state_dict(orginal_state_list[remain_layer_list[1][j] + num_of_block[0] + 2])
        else:
            child.load_state_dict(orginal_state_list[remain_layer_list[1][j] + num_of_block[0] + 2])
        j += 1

    for child in pruned_model.layer3.children():
        if k == 0:
            if remain_layer_list[2][k] != 0:
                print('The first layer in layer 3 is initialized by random')
            else:
                child.load_state_dict(orginal_state_list[remain_layer_list[2][k] + num_of_block[0] + num_of_block[1] + 2])
        else:
            child.load_state_dict(orginal_state_list[remain_layer_list[2][k] + num_of_block[0] + num_of_block[1] + 2])
        k += 1
    for child in pruned_model.layer4.children():
        if l == 0:
            if remain_layer_list[2][l] != 0:
                print('The first layer in layer 4 is initialized by random')
            else:
                child.load_state_dict(orginal_state_list[remain_layer_list[2][l] + num_of_block[0] + num_of_block[1] + num_of_block[2] + 2])
        else:
            child.load_state_dict(orginal_state_list[remain_layer_list[2][l] + num_of_block[0] + num_of_block[1] + num_of_block[2] + 2])
        l += 1

    pruned_model.fc.load_state_dict(orginal_state_list[-1])
    return pruned_model


def load_vgg_layer_params(orginal_model, orginal_conv_list, pruned_model, pruned_conv_list):
    orginal_state_list = []
    num_conv = len(orginal_conv_list[0])
    i = 0
    j = 0
    for child in orginal_model.features.children():
        if i == orginal_conv_list[0][0]:
            del orginal_conv_list[0][0]
            orginal_state_list.append(child.state_dict())
            if orginal_conv_list[0] == []:
                break

        i += 1

    for child in orginal_model.classifier.children():
        if j == orginal_conv_list[1][0]:
            del orginal_conv_list[1][0]
            orginal_state_list.append(child.state_dict())
            if orginal_conv_list[1] == []:
                break

        j += 1

    count = 0
    print('These layers are initialized with the pretrained model')
    for child in pruned_model.features.children():
        if count == pruned_conv_list[0][0][0]:
            if pruned_conv_list[0][0][1] != -1:
                print(child)
                child.load_state_dict(orginal_state_list[pruned_conv_list[0][0][1]])
            else:
                print('This layer is initialized by random')

            del pruned_conv_list[0][0]
            if pruned_conv_list[0] == []:
                break

        count += 1

    linear_count = 0
    for child in pruned_model.classifier.children():
        if linear_count == pruned_conv_list[1][0][0]:
            if pruned_conv_list[1][0][1] != -1:
                print(child)
                child.load_state_dict(orginal_state_list[pruned_conv_list[1][0][1] + 16])  # vgg19 has 16 conv layers
            else:
                print('This layer is initialized by random')

            del pruned_conv_list[1][0]
            if pruned_conv_list[1] == []:
                break

        linear_count += 1

    return pruned_model


if __name__ == '__main__':
    model = Iresnet50_cut3()
    orginal_model = models.resnet50(pretrained=False)
    ckpt = torch.load("./pretrained_model/resnet50.pth")
    orginal_model.load_state_dict(ckpt)
    orginal_state_list = get_Iresnet_layer_params(orginal_model)