import test_path_setting
import torch
import torchvision.models as models


def model_size_test():
    print('alexnet')
    model = models.alexnet()
    total_params = sum(param.numel() for param in model.parameters())
    print(total_params)

    print('vgg11')
    model = models.vgg11()
    total_params = sum(param.numel() for param in model.parameters())
    print(total_params)

    print('resnet18')
    model = models.resnet18()
    total_params = sum(param.numel() for param in model.parameters())
    print(total_params)

    print('squeezenet1_0')
    model = models.squeezenet1_0()
    total_params = sum(param.numel() for param in model.parameters())
    print(total_params)

    print('squeezenet1_1')
    model = models.squeezenet1_1()
    print(model)
    inputs = torch.randn(16, 3, 128, 128)
    outputs = model(inputs)
    print(outputs.size())
    outputs = model.features(inputs)
    print(outputs.size())
    inputs = torch.randn(16, 3, 256, 256)
    outputs = model(inputs)
    print(outputs.size())
    outputs = model.features(inputs)
    print(outputs.size())
    total_params = sum(param.numel() for param in model.parameters())
    print(total_params)

    print('densenet121')
    model = models.densenet121()
    total_params = sum(param.numel() for param in model.parameters())
    print(total_params)

    print('inception_v3')
    model = models.inception_v3()
    total_params = sum(param.numel() for param in model.parameters())
    print(total_params)

    print('googlenet')
    model = models.googlenet()
    total_params = sum(param.numel() for param in model.parameters())
    print(total_params)

    print('shufflenet_v2_x0_5')
    model = models.shufflenet_v2_x0_5()
    total_params = sum(param.numel() for param in model.parameters())
    print(total_params)

    print('mobilenet_v2')
    model = models.mobilenet_v2()
    total_params = sum(param.numel() for param in model.parameters())
    print(total_params)

    print('resnext50_32x4d')
    model = models.resnext50_32x4d()
    total_params = sum(param.numel() for param in model.parameters())
    print(total_params)

    print('wide_resnet50_2')
    model = models.wide_resnet50_2()
    total_params = sum(param.numel() for param in model.parameters())
    print(total_params)

    print('mnasnet0_5')
    model = models.mnasnet0_5()
    total_params = sum(param.numel() for param in model.parameters())
    print(total_params)


if __name__ == "__main__":
    model_size_test()
