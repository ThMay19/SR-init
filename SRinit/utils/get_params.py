from collections import OrderedDict
from args import args


def get_cresnet_layer_params(target_model):
    orginal_state_list = []
    orginal_state_list.append(target_model.conv1.state_dict())
    orginal_state_list.append(target_model.bn1.state_dict())
    for child in target_model.layer1.children():
        orginal_state_list.append(child.state_dict())

    for child in target_model.layer2.children():
        orginal_state_list.append(child.state_dict())

    for child in target_model.layer3.children():
        orginal_state_list.append(child.state_dict())

    orginal_state_list.append(target_model.linear.state_dict())
    return orginal_state_list


def get_cvgg_layer_params(target_model):
    error = OrderedDict()
    original_state_list = []
    for child in target_model.features.children():
        if child.state_dict() != error:
            original_state_list.append(child.state_dict())
    for child in target_model.classifier.children():
        if child.state_dict() != error:
            original_state_list.append(child.state_dict())
    return original_state_list


def get_Iresnet_layer_params(target_model):
    orginal_state_list = []
    orginal_state_list.append(target_model.conv1.state_dict())
    orginal_state_list.append(target_model.bn1.state_dict())
    for child in target_model.layer1.children():
        orginal_state_list.append(child.state_dict())

    for child in target_model.layer2.children():
        orginal_state_list.append(child.state_dict())

    for child in target_model.layer3.children():
        orginal_state_list.append(child.state_dict())

    for child in target_model.layer4.children():
        orginal_state_list.append(child.state_dict())
    return orginal_state_list


def get_layer_params(model):
    if args.arch == 'resnet56':
        return get_cresnet_layer_params(model)
    elif args.arch == 'resnet20':
        return get_cresnet_layer_params(model)
    elif args.arch == 'cvgg16_bn':
        return get_cvgg_layer_params(model)
    elif args.arch == 'ResNet50':
        return get_Iresnet_layer_params(model)
    elif args.arch == 'ResNet101':
        return get_Iresnet_layer_params(model)
    elif args.arch == 'resnet110':
        return get_cresnet_layer_params(model)