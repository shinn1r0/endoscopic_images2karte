import test_path_setting
import torch
from models import lrcn


def lrcn_test(num_classes, parameter):
    model = lrcn(num_classes, 60)
    print(model)
    if parameter:
        for name, param in model.named_parameters():
            print(name, end="")
            print(":", param.numel())
    total_params = sum(param.numel() for param in model.parameters())
    print("total_params:", total_params)

    # inputs = torch.randn(5, 60, 3, 128, 128)
    inputs = torch.randn(5, 10)
    print(inputs.size())
    outputs = model(inputs)
    print(outputs.size())


if __name__ == "__main__":
    try:
        num_classes = int(input("num_classes: "))
    except ValueError:
        num_classes = 140
    try:
        parameter = bool(input("parameter: "))
    except ValueError:
        parameter = False
    try:
        expansion = bool(input("expansion: "))
    except ValueError:
        expansion = False

    lrcn_test(num_classes, parameter)
