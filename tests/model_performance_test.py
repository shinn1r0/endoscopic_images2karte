import copy
import os
import time
from pathlib import Path
from pprint import pprint

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn import metrics

import test_path_setting
from dataset import Images2KarteDataset, ImagesSeq2KarteDataset
from models import simple_model1, simple_model2, densenet, resnet3d
import transforms_3d


def dataloader(dataset_name, transform=None, batch_size=1, num_worker=0, seq=False, datasets_path=None):
    dataset_path = str(datasets_path / dataset_name)
    os.makedirs(dataset_path, exist_ok=True)
    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
        # testset = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform)
    elif dataset_name == "kinetics400":
        dataset = datasets.Kinetics400(root=dataset_path, frames_per_clip=5, transform=transform)

    dataset_size = len(dataset)
    train_dataset_size = int(dataset_size * 0.7)
    val_dataset_size = int(dataset_size * 0.15)
    test_dataset_size = dataset_size - (train_dataset_size + val_dataset_size)

    train_dataset, val_dataset, test_dataset = \
        random_split(dataset, [train_dataset_size, val_dataset_size, test_dataset_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    dataloader_dict = {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader
    }

    return dataloader_dict


def train_model(model, dataloaders, criterion, optimizer, num_classes, num_epochs, device,
                output_path, train_result_txt_path, train_result_dict_path):
    """
    Train Model

    Args:
        model (torch.nn.modules.module.Module):
        dataloaders (dict[str, torch.util.data.Dataloader]):
        criterion (torch.nn.modules.loss._Loss):
        optimizer (torch.optim.optimizer.Optimizer):
        num_classes (int):
        num_epochs (int):
        device (torch.device):
        output_path (str):
        train_result_txt_path (str):
        train_result_dict_path (str):

    Returns:
        torch.nn.modules.module.Module, list[int]
    """
    since = time.time()
    epoch_loss_list = list()
    epoch_micro_metrics_list = list()
    epoch_macro_metrics_list = list()
    epoch_acc_list = list()
    epoch_all_acc_list = list()
    train_result_dict = {
        'epoch_loss_list': epoch_loss_list,
        'epoch_micro_metrics_list': epoch_micro_metrics_list,
        'epoch_macro_metrics_list': epoch_macro_metrics_list,
        'epoch_acc_list': epoch_acc_list,
        'epoch_all_acc_list': epoch_all_acc_list
    }
    val_acc_history = list()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = torch.tensor(0.0)
            running_micro_metrics = 0.0
            running_macro_metrics = 0.0
            running_corrects = torch.tensor(0.0)
            running_all_corrects = torch.tensor(0.0)
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = (outputs > 0.5).float()
                    corrects = torch.eq(preds, labels.data).float().sum(dim=1) / num_classes

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_micro_metrics += metrics.f1_score(labels.data.cpu(), preds.cpu(), average="micro")
                running_macro_metrics += metrics.f1_score(labels.data.cpu(), preds.cpu(), average="macro")
                running_corrects += torch.sum(corrects)
                running_all_corrects += torch.sum(corrects.int())
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_micro_metrics = running_micro_metrics / len(dataloaders[phase].dataset)
            epoch_macro_metrics = running_macro_metrics / len(dataloaders[phase].dataset)
            epoch_acc = (running_corrects.double() / len(dataloaders[phase].dataset)).item()
            epoch_all_acc = (running_all_corrects.double() / len(dataloaders[phase].dataset)).item()

            train_result_dict['epoch_loss_list'].append(epoch_loss)
            train_result_dict['epoch_micro_metrics_list'].append(epoch_micro_metrics)
            train_result_dict['epoch_macro_metrics_list'].append(epoch_macro_metrics)
            train_result_dict['epoch_acc_list'].append(epoch_acc)
            train_result_dict['epoch_all_acc_list'].append(epoch_all_acc)
            with open(train_result_dict_path, mode="wb") as f:
                joblib.dump(train_result_dict, f, compress=3)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} ALLAcc: {epoch_all_acc:.4f}')
            print(f'Metrics Micro: {epoch_micro_metrics} Metrics Macro: {epoch_macro_metrics}')
            with open(train_result_txt_path, mode="a") as f:
                pprint(f'Epoch: {epoch}', stream=f)
                pprint(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} ALLAcc: {epoch_all_acc:.4f}', stream=f)
                pprint(f'Metrics Micro: {epoch_micro_metrics} Metrics Macro: {epoch_macro_metrics}', stream=f)

            if epoch:
                previous_model_path = str(output_path / ('model-epoch' + str(epoch-1)))
                os.remove(previous_model_path)
            model_path = str(output_path / ('model-epoch' + str(epoch)))
            torch.save(model.state_dict(), model_path)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = model.state_dict()
                best_model_path = str(output_path / ('best_model-epoch' + str(epoch)))
                torch.save(best_model, best_model_path)
                best_model_wts = copy.deepcopy(best_model)
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        print()
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def main(batch_size, epoch, dataset_name, model_name, seq=False):
    print('loading dataset & creating dataloaders')
    file_path = Path(__file__)
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()

    if dataset_name == "cifar10":
        num_classes = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif dataset_name == "kinetics400":
        num_classes = 400
        transform = transforms.Compose([
            transforms_3d.ToFloatTensorInZeroOne(),
            transforms_3d.Resize((128, 171)),
            transforms_3d.RandomHorizontalFlip(),
            transforms_3d.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                    std=[0.22803, 0.22145, 0.216989]),
            transforms_3d.RandomCrop((112, 112))
        ])
    dataloaders = dataloader(dataset_name=dataset_name, transform=transform,
                             batch_size=batch_size, num_worker=0, seq=seq, datasets_path=datasets_path)
    print("creating model")
    if model_name == 'simple_model1':
        model = simple_model1(num_classes)
    elif model_name == 'simple_model2':
        model = simple_model2(num_classes)
    elif model_name == 'densenet':
        model = densenet(num_classes)
    elif model_name == 'resnet3d':
        model = resnet3d(num_classes)
    else:
        exit(1)

    print("cuda settings")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print('optimizer and criterion setting')
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters())
    # criterion = nn.MultiLabelSoftMarginLoss()
    weight = torch.ones([num_classes]).to(device)
    criterion = nn.BCEWithLogitsLoss(weight=weight)

    print('discription setting')
    outputs_path = (file_path / '..' / '..' / 'outputs').resolve()
    output_dir = model_name + '-' + dataset_name
    output_path = outputs_path / output_dir
    output_dir_index = 0
    while os.path.isdir(output_path):
        output_dir_index += 1
        output_path = outputs_path / (output_dir + '-' + str(output_dir_index))
    os.makedirs(output_path, exist_ok=True)

    train_discription_path = str(output_path / 'train_discription.txt')
    with open(train_discription_path, mode="w") as f:
        pprint(model_name, stream=f)
        pprint(dataset_name, stream=f)
        pprint('batch_size: ' + str(batch_size), stream=f)
        pprint('epoch: ' + str(epoch), stream=f)

        pprint('transform', stream=f)
        pprint(transform, stream=f)

        pprint('criterion', stream=f)
        pprint(criterion, stream=f)

        pprint('optimizer', stream=f)
        pprint(optimizer, stream=f)

    print('start training')
    train_result_txt_path = str(output_path / 'train_result.txt')
    train_result_dict_path = str(output_path / 'train_result.joblib')
    model, hist = train_model(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer,
                              num_classes=num_classes, num_epochs=epoch, device=device,
                              output_path=output_path,
                              train_result_txt_path=train_result_txt_path,
                              train_result_dict_path=train_result_dict_path)
    print(hist)


if __name__ == "__main__":
    # main(batch_size=64, epoch=50, dataset_name='dataset1', model_name='simple_model2', seq=False)
    main(batch_size=32, epoch=50, dataset_name='cifar10', model_name='densenet', seq=False)
    # main(batch_size=24, epoch=45, dataset_name='kinetics400', model_name='resnet3d', seq=True)
