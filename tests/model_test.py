import test_path_setting
import torch
from models import mycnn, densenet_121, densenet_169, densenet_201, densenet_161, squeezenet, resnet3d, lrcn


def mycnn_test(num_classes, parameter):
    model = mycnn(num_classes)
    print(model)
    if parameter:
        for name, param in model.named_parameters():
            print(name, end="")
            print(":", param.numel())
    total_params = sum(param.numel() for param in model.parameters())
    print("total_params:", total_params)
    print(type(model))


def densenet_121_test(num_classes, parameter, expansion):
    model = densenet_121(num_classes, expansion)
    print(model)
    if parameter:
        for name, param in model.named_parameters():
            print(name, end="")
            print(":", param.numel())
    total_params = sum(param.numel() for param in model.parameters())
    print("total_params:", total_params)
    print(type(model))


def densenet_169_test(num_classes, parameter, expansion):
    model = densenet_169(num_classes)
    print(model)
    if parameter:
        for name, param in model.named_parameters():
            print(name, end="")
            print(":", param.numel())
    total_params = sum(param.numel() for param in model.parameters())
    print("total_params:", total_params)
    print(type(model))


def densenet_201_test(num_classes, parameter, expansion):
    model = densenet_201(num_classes)
    print(model)
    if parameter:
        for name, param in model.named_parameters():
            print(name, end="")
            print(":", param.numel())
    total_params = sum(param.numel() for param in model.parameters())
    print("total_params:", total_params)
    print(type(model))


def densenet_161_test(num_classes, parameter, expansion):
    model = densenet_161(num_classes)
    print(model)
    if parameter:
        for name, param in model.named_parameters():
            print(name, end="")
            print(":", param.numel())
    total_params = sum(param.numel() for param in model.parameters())
    print("total_params:", total_params)
    print(type(model))


def squeezenet_test(num_classes, parameter, expansion):
    model = squeezenet(num_classes)
    print(model)
    if parameter:
        for name, param in model.named_parameters():
            print(name, end="")
            print(":", param.numel())
    total_params = sum(param.numel() for param in model.parameters())
    print("total_params:", total_params)
    print(type(model))
    inputs = torch.randn(5, 3, 256, 256)
    print(inputs.size())
    outputs = model(inputs)
    print(outputs.size())


def resnet3d_test(num_classes, parameter):
    model = resnet3d(num_classes)
    print(model)
    if parameter:
        for name, param in model.named_parameters():
            print(name, end="")
            print(":", param.numel())
    total_params = sum(param.numel() for param in model.parameters())
    print("total_params:", total_params)
    print(type(model))


def lrcn_test(num_classes, time_steps, parameter):
    model = lrcn(num_classes, time_steps)
    print(model)
    if parameter:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print('gradient' + name, end="")
                print(":", param.numel())
            else:
                print(name, end="")
                print(":", param.numel())
    total_params = sum(param.numel() for param in model.parameters())
    print("total_params:", total_params)
    print(type(model))


if __name__ == "__main__":
    try:
        num_classes = int(input("num_classes: "))
    except ValueError:
        num_classes = 140
    try:
        time_steps = int(input("time_steps: "))
    except ValueError:
        time_steps = 60
    try:
        parameter = bool(input("parameter: "))
    except ValueError:
        parameter = False
    try:
        expansion = bool(input("expansion: "))
    except ValueError:
        expansion = False

    model_name = input("model_name: ")
    if model_name == "mycnn":
        mycnn_test(num_classes, parameter)
    elif model_name == "resnet3d":
        resnet3d_test(num_classes, parameter)
    elif model_name == "lrcn":
        lrcn_test(num_classes, time_steps, parameter)
    elif model_name == "densenet121":
        densenet_121_test(num_classes, parameter, expansion)
    elif model_name == "densenet169":
        densenet_169_test(num_classes, parameter, expansion)
    elif model_name == "densenet201":
        densenet_201_test(num_classes, parameter, expansion)
    elif model_name == "densenet161":
        densenet_161_test(num_classes, parameter, expansion)
    elif model_name == "squeezenet":
        squeezenet_test(num_classes, parameter, expansion)
