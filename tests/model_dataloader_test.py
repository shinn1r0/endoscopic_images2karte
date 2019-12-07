import copy
import time
import joblib
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from images2karte.dataset import Images2KarteDataset
from images2karte.models import simple_model2


def dataloader(dataset_file, transform, batch_size=1, num_worker=0):
    """
    Load Data

    Args:
        dataset_file (str):
        transform (torchvision.transforms.transforms.Compose):
        batch_size (int):
        num_worker (int):

    Returns:
        dict[str, torch.util.data.Dataloader]

    """
    dataset = Images2KarteDataset(dataset_file, transform=transform)

    train_dataset_size = int(len(dataset) * 0.7)
    val_dataset_size = int(len(dataset) * 0.15)
    test_dataset_size = len(dataset) - (train_dataset_size + val_dataset_size)

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


def train_model(model, dataloaders, criterion, optimizer, num_classes, num_epochs, device):
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

    Returns:
        torch.nn.modules.module.Module, list[int]
    """
    since = time.time()
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
            running_corrects = torch.tensor(0.0)
            running_all_corrects = torch.tensor(0.0)
            # count = 0
            for inputs, labels in tqdm(dataloaders[phase]):
                # print(count)
                # count += 1
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
                running_corrects += torch.sum(corrects)
                running_all_corrects += torch.sum(corrects.int())
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_all_acc = running_all_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} ALLAcc: {epoch_all_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        print()
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def main():
    file_path = Path(__file__)
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / 'image-labels-ver4'
    dataset_file = str(dataset_path / 'data.joblib')
    labels_path = datasets_path / 'labels'
    label_file = str(labels_path / 'labels2list.joblib')
    with open(label_file, mode="rb") as f:
        label_list = joblib.load(f)

    transform = transforms.Compose([
        # transforms.Resize((480, 560)),
        # transforms.CenterCrop((448, 512)),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    batch_size = 16
    dataloaders = dataloader(dataset_file=dataset_file, transform=transform, batch_size=batch_size)
    num_classes = len(label_list)
    model = simple_model2(num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.MultiLabelSoftMarginLoss()

    for inputs, labels in dataloaders["train"]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        print(outputs.size())
        break
    # model, hist = train_model(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer,
    #                           num_classes=num_classes, num_epochs=25, device=device)
    # print(hist)


if __name__ == "__main__":
    main()
