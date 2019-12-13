from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.models import densenet121, densenet169, densenet201, densenet161, squeezenet1_1
from torchvision.models.video import r2plus1d_18


def densenet_121(num_classes, expansion=False):
    """

    Args:
        num_classes (int):

    Returns:
        torch.nn.modules.module.Module

    """

    model = densenet121(pretrained=False, progress=True)
    num_features = model.classifier.in_features
    if expansion:
        model.classifier = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(in_features=num_features, out_features=200)),
            ('norm1', nn.BatchNorm1d(num_features=200)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.25)),
            ('last', nn.Linear(in_features=200, out_features=num_classes))
        ]))
    else:
        model.classifier = nn.Linear(num_features, num_classes, bias=True)

    return model


def densenet_169(num_classes, expansion=False):
    """

    Args:
        num_classes (int):

    Returns:
        torch.nn.modules.module.Module

    """

    model = densenet169(pretrained=False, progress=True)
    num_features = model.classifier.in_features
    if expansion:
        model.classifier = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(in_features=num_features, out_features=1000)),
            ('norm1', nn.BatchNorm1d(num_features=1000)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.25)),
            ('dense2', nn.Linear(in_features=1000, out_features=200)),
            ('norm2', nn.BatchNorm1d(num_features=200)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.25)),
            ('last', nn.Linear(in_features=200, out_features=num_classes))
        ]))
    else:
        model.classifier = nn.Linear(num_features, num_classes, bias=True)

    return model


def densenet_201(num_classes, expansion=False):
    """

    Args:
        num_classes (int):

    Returns:
        torch.nn.modules.module.Module

    """

    model = densenet201(pretrained=False, progress=True)
    num_features = model.classifier.in_features
    if expansion:
        model.classifier = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(in_features=num_features, out_features=1000)),
            ('norm1', nn.BatchNorm1d(num_features=1000)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.25)),
            ('dense2', nn.Linear(in_features=1000, out_features=200)),
            ('norm2', nn.BatchNorm1d(num_features=200)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.25)),
            ('last', nn.Linear(in_features=200, out_features=num_classes))
        ]))
    else:
        model.classifier = nn.Linear(num_features, num_classes, bias=True)

    return model


def densenet_161(num_classes, expansion=False):
    """

    Args:
        num_classes (int):

    Returns:
        torch.nn.modules.module.Module

    """

    model = densenet161(pretrained=False, progress=True)
    num_features = model.classifier.in_features
    if expansion:
        model.classifier = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(in_features=num_features, out_features=1000)),
            ('norm1', nn.BatchNorm1d(num_features=1000)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.25)),
            ('dense2', nn.Linear(in_features=1000, out_features=200)),
            ('norm2', nn.BatchNorm1d(num_features=200)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.25)),
            ('last', nn.Linear(in_features=200, out_features=num_classes))
        ]))
    else:
        model.classifier = nn.Linear(num_features, num_classes, bias=True)

    return model


def squeezenet(num_classes):
    """

    Args:
        num_classes (int):

    Returns:
        torch.nn.modules.module.Module

    """

    model = nn.Sequential(OrderedDict([
        ('squeezenet', squeezenet1_1(pretrained=False, progress=True)),
        ('dense', nn.Linear(in_features=1000, out_features=200)),
        ('norm', nn.BatchNorm1d(num_features=200)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.25)),
        ('last', nn.Linear(in_features=200, out_features=num_classes))
    ]))

    return model


def resnet3d(num_classes, expansion=False, maxpool=False):
    """

    Args:
        num_classes (int):

    Returns:
        torch.nn.modules.module.Module

    """

    model = r2plus1d_18(pretrained=False, progress=True)
    num_features = model.fc.in_features
    if expansion:
        model.fc = nn.Sequential(OrderedDict([
            ('dense', nn.Linear(in_features=num_features, out_features=200)),
            ('norm', nn.BatchNorm1d(num_features=200)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.25)),
            ('last', nn.Linear(in_features=200, out_features=num_classes))
        ]))
    else:
        model.fc = nn.Linear(num_features, num_classes, bias=True)
    if maxpool:
        model.avgpool = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))

    return model


def mycnn(num_classes):
    """

    Args:
        num_classes (int):

    Returns:
        torch.nn.modules.module.Module

    """
    class Flatten(nn.Module):
        def __init__(self):
            super(Flatten, self).__init__()

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return x

    model = nn.Sequential(OrderedDict([
        ('conv00', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))),
        ('norm00', nn.BatchNorm2d(num_features=32)),
        ('relu00', nn.ReLU(inplace=False)),
        ('conv01', nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3))),
        ('norm01', nn.BatchNorm2d(num_features=32)),
        ('relu01', nn.ReLU(inplace=False)),
        ('pool0', nn.MaxPool2d(kernel_size=(2, 2))),

        ('conv10', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))),
        ('norm10', nn.BatchNorm2d(num_features=64)),
        ('relu10', nn.ReLU(inplace=False)),
        ('conv11', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))),
        ('norm11', nn.BatchNorm2d(num_features=64)),
        ('relu11', nn.ReLU(inplace=False)),
        ('pool1', nn.MaxPool2d(kernel_size=(2, 2))),

        ('conv20', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))),
        ('norm20', nn.BatchNorm2d(num_features=128)),
        ('relu20', nn.ReLU(inplace=False)),
        ('conv21', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3))),
        ('norm21', nn.BatchNorm2d(num_features=128)),
        ('relu21', nn.ReLU(inplace=False)),
        ('pool2', nn.MaxPool2d(kernel_size=(2, 2))),

        ('conv30', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3))),
        ('norm30', nn.BatchNorm2d(num_features=256)),
        ('relu30', nn.ReLU(inplace=False)),
        ('conv31', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3))),
        ('norm31', nn.BatchNorm2d(num_features=256)),
        ('relu31', nn.ReLU(inplace=False)),
        ('pool3', nn.MaxPool2d(kernel_size=(2, 2))),

        ('conv40', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3))),
        ('norm40', nn.BatchNorm2d(num_features=512)),
        ('relu40', nn.ReLU(inplace=False)),
        ('conv41', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3))),
        ('norm41', nn.BatchNorm2d(num_features=512)),
        ('relu41', nn.ReLU(inplace=False)),
        ('pool4', nn.MaxPool2d(kernel_size=(2, 2))),

        ('flatten', Flatten()),
        ('dense', nn.Linear(in_features=8192, out_features=1000, bias=True)),
        ('norm', nn.BatchNorm1d(num_features=1000)),
        ('relu', nn.ReLU(inplace=False)),
        ('dropout', nn.Dropout(p=0.2, inplace=False)),
        ('last', nn.Linear(in_features=1000, out_features=num_classes, bias=True))
    ]))
    return model


def lrcn(num_classes, lrcn_time_steps, lstm_hidden_size=200, lstm_num_layers=2):
    """

    Args:
        num_classes (int):

    Returns:
        torch.nn.modules.module.Module
    """
    class TimeDistributed(nn.Module):
        def __init__(self, layer, time_steps):
            super(TimeDistributed, self).__init__()
            # self.layers = nn.ModuleList([layer for _ in range(time_steps)])
            self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(time_steps)])

        def forward(self, x):
            batch_size, time_steps, *_ = x.size()
            # outputs = list()
            for i, layer in enumerate(self.layers):
                x = layer(x)
#                 output_t = layer(x[:, i])
                # if i == 0:
                    # output = output_t.unsqueeze(1)
                # else:
#                     output = torch.cat((output, output_t.unsqueeze(1)), 1)
                # outputs.append(output_t)
            # output = torch.stack(outputs, dim=1)
            # return output
            return x

    class BiLSTMHidden2Dense(nn.Module):
        def __init__(self):
            super(BiLSTMHidden2Dense, self).__init__()

        def forward(self, x):
            lstm_output, (hn, cn) = x
            lstm_last_hidden_state = hn[-2:].transpose(0, 1).contiguous().view(hn.size(1), -1)
            return lstm_last_hidden_state

    cnn_model = squeezenet1_1(pretrained=False, progress=True)
    model = nn.Sequential(OrderedDict([
        ('timedistributed_cnn', TimeDistributed(nn.Conv2d(3, 60, (1, 1)), time_steps=lrcn_time_steps)),
        # ('timedistributed_cnn', TimeDistributed(cnn_model, time_steps=lrcn_time_steps)),
#         ('bidirectional_stacked_lstm', nn.LSTM(input_size=1000, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers,
                                               # batch_first=True, dropout=0.2, bidirectional=True)),
        # ('hidden2dense', BiLSTMHidden2Dense()),
        # ('dense', nn.Linear(in_features=2*lstm_hidden_size, out_features=lstm_hidden_size)),
        # ('norm', nn.BatchNorm1d(num_features=lstm_hidden_size)),
        # ('relu', nn.ReLU()),
        # ('dropout', nn.Dropout(p=0.25)),
#         ('last', nn.Linear(in_features=lstm_hidden_size, out_features=num_classes))
    ]))

    return model
