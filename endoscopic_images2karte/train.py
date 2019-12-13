import argparse
import copy
import os
import time
from pathlib import Path
from pprint import pprint
import shutil

import joblib
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from dataset import Images2KarteTrainDataset, Images2KarteTestDataset, ImagesSeq2KarteDataset
from models import mycnn, densenet_121, densenet_169, densenet_201, densenet_161, squeezenet, resnet3d, lrcn
from utils import calc_f1score
from transforms_3d import Normalize


def dataloader(dataset_file, dataloaders_path, test_dataloader_path, transform=None, transform_3d=None,
               image_num_limit=None, limit=None, batch_size=1, num_worker=0, seq=False, channel_first=False):
    """
    Load Data

    Args:
        dataset_file (str):
        transform (torchvision.transforms.transforms.Compose):
        transform_3d (torchvision.transforms.transforms.Compose):
        limit (int):
        batch_size (int):
        num_worker (int):

    Returns:
        dict[str, torch.util.data.Dataloader]

    """
    if seq:
        dataset = ImagesSeq2KarteDataset(dataset_file, transform=transform, transform_3d=transform_3d,
                                         image_num_limit=image_num_limit, limit=limit, channel_first=channel_first)
        dataset_size = len(dataset)
        train_dataset_size = int(dataset_size * 0.7)
        val_dataset_size = int(dataset_size * 0.15)
        test_dataset_size = dataset_size - (train_dataset_size + val_dataset_size)

        train_dataset, val_dataset, test_dataset = \
            random_split(dataset, [train_dataset_size, val_dataset_size, test_dataset_size])
    else:
        with open(dataset_file, mode="rb") as f:
            data_dict = joblib.load(f)
        image_paths = data_dict["image_paths"]
        labels_list = data_dict["label"]

        dataset_size = len(image_paths)
        train_size = int(dataset_size * 0.7)
        val_size = int(dataset_size * 0.15)
        id_all = np.random.choice(dataset_size, dataset_size, replace=False)
        id_train = id_all[:train_size+val_size]
        id_test = id_all[train_size+val_size:]
        train_image_paths = [x for (i, x) in enumerate(image_paths) if i in id_train]
        test_image_paths = [x for (i, x) in enumerate(image_paths) if i in id_test]
        train_labels_list = [x for (i, x) in enumerate(labels_list) if i in id_train]
        test_labels_list = [x for (i, x) in enumerate(labels_list) if i in id_test]
        train_data_dict = {"image_paths": train_image_paths, "label": train_labels_list}
        test_data_dict = {"image_paths": test_image_paths, "label": test_labels_list}
        train_val_dataset = Images2KarteTrainDataset(train_data_dict, transform=transform, limit=limit)
        test_dataset = Images2KarteTestDataset(test_data_dict, transform=transform, limit=limit)
        train_dataset_size = int(len(train_val_dataset) * 0.82)
        val_dataset_size = len(train_val_dataset) - train_dataset_size
        train_dataset, val_dataset = random_split(train_val_dataset, [train_dataset_size, val_dataset_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    if seq:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    else:
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_worker)
    with open(test_dataloader_path, mode="wb") as f:
        joblib.dump(test_dataloader, f, compress=3)

    dataloader_dict = {
        "train": train_dataloader,
        "val": val_dataloader,
    }
    with open(dataloaders_path, mode="wb") as f:
        joblib.dump(dataloader_dict, f, compress=3)

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
    train_result_dict = {
        'epoch_loss_list': list(),
        'epoch_total_micro_precision_list': list(),
        'epoch_total_micro_recall_list': list(),
        'epoch_total_micro_f1score_list': list(),
        'epoch_total_macro_precision_list': list(),
        'epoch_total_macro_recall_list': list(),
        'epoch_total_macro_f1score_list': list(),
        'epoch_total_acc_list': list(),
        'epoch_total_all_acc_list': list(),
        'epoch_abnormality_precision_list': list(),
        'epoch_abnormality_recall_list': list(),
        'epoch_abnormality_f1score_list': list(),
        'epoch_abnormality_acc_list': list(),
        'epoch_label_micro_precision_list': list(),
        'epoch_label_micro_recall_list': list(),
        'epoch_label_micro_f1score_list': list(),
        'epoch_label_macro_precision_list': list(),
        'epoch_label_macro_recall_list': list(),
        'epoch_label_macro_f1score_list': list(),
        'epoch_label_acc_list': list(),
        'epoch_label_all_acc_list': list()
    }
    val_f1score_history = list()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1score = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('=' * 80)
        with open(train_result_txt_path, mode="a") as f:
            pprint(f'Epoch {epoch}/{num_epochs - 1}', stream=f)
            pprint('=' * 80, stream=f)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            dataset_length = len(dataloaders[phase].dataset)
            running_loss = torch.tensor(0.0)
            running_mcm = np.zeros([num_classes, 2, 2], dtype=np.int)
            running_total_corrects = torch.tensor(0.0)
            running_total_all_corrects = torch.tensor(0.0)
            running_abnormality_corrects = torch.tensor(0.0)
            running_label_corrects = torch.tensor(0.0)
            running_label_all_corrects = torch.tensor(0.0)
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = (outputs > 0.5).float()
                    total_corrects = torch.eq(preds, labels.data).float().sum(dim=1) / num_classes
                    abnormality_corrects = torch.eq(preds[:, 0], labels.data[:, 0]).int()
                    label_corrects = torch.eq(preds[:, 1:], labels.data[:, 1:]).float().sum(dim=1) / (num_classes - 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_mcm += metrics.multilabel_confusion_matrix(labels.data.cpu(), preds.cpu())
                running_loss += loss.item() * inputs.size(0)
                running_total_corrects += torch.sum(total_corrects)
                running_total_all_corrects += torch.sum(total_corrects.int())
                running_abnormality_corrects += torch.sum(abnormality_corrects)
                running_label_corrects += torch.sum(label_corrects)
                running_label_all_corrects += torch.sum(label_corrects.int())
            total_f1score_dict = calc_f1score(running_mcm, multi=True)
            abnormality_f1score_dict = calc_f1score(running_mcm[0], multi=False)
            label_f1score_dict = calc_f1score(running_mcm[1:], multi=True)
            epoch_loss = running_loss / dataset_length
            epoch_total_acc = (running_total_corrects.double() / dataset_length).item()
            epoch_total_all_acc = (running_total_all_corrects.double() / dataset_length).item()
            epoch_abnormality_acc = (running_abnormality_corrects.double() / dataset_length).item()
            epoch_label_acc = (running_label_corrects.double() / dataset_length).item()
            epoch_label_all_acc = (running_label_all_corrects.double() / dataset_length).item()

            train_result_dict['epoch_loss_list'].append(epoch_loss)
            train_result_dict['epoch_total_micro_precision_list'].append(total_f1score_dict['micro_precision'])
            train_result_dict['epoch_total_micro_recall_list'].append(total_f1score_dict['micro_recall'])
            train_result_dict['epoch_total_micro_f1score_list'].append(total_f1score_dict['micro_f1score'])
            train_result_dict['epoch_total_macro_precision_list'].append(total_f1score_dict['macro_precision'])
            train_result_dict['epoch_total_macro_recall_list'].append(total_f1score_dict['macro_recall'])
            train_result_dict['epoch_total_macro_f1score_list'].append(total_f1score_dict['macro_f1score'])
            train_result_dict['epoch_total_acc_list'].append(epoch_total_acc)
            train_result_dict['epoch_total_all_acc_list'].append(epoch_total_all_acc)
            train_result_dict['epoch_abnormality_precision_list'].append(abnormality_f1score_dict['precision'])
            train_result_dict['epoch_abnormality_recall_list'].append(abnormality_f1score_dict['recall'])
            train_result_dict['epoch_abnormality_f1score_list'].append(abnormality_f1score_dict['f1score'])
            train_result_dict['epoch_abnormality_acc_list'].append(epoch_abnormality_acc)
            train_result_dict['epoch_label_micro_precision_list'].append(label_f1score_dict['micro_precision'])
            train_result_dict['epoch_label_micro_recall_list'].append(label_f1score_dict['micro_recall'])
            train_result_dict['epoch_label_micro_f1score_list'].append(label_f1score_dict['micro_f1score'])
            train_result_dict['epoch_label_macro_precision_list'].append(label_f1score_dict['macro_precision'])
            train_result_dict['epoch_label_macro_recall_list'].append(label_f1score_dict['macro_recall'])
            train_result_dict['epoch_label_macro_f1score_list'].append(label_f1score_dict['macro_f1score'])
            train_result_dict['epoch_label_acc_list'].append(epoch_label_acc)
            train_result_dict['epoch_label_all_acc_list'].append(epoch_label_all_acc)
            with open(train_result_dict_path, mode="wb") as f:
                joblib.dump(train_result_dict, f, compress=3)

            print(f'{phase}')
            print('-' * 70)
            print(f'Loss: {epoch_loss:.4f}')
            print('Total' + '-'*46)
            print(f'Acc: {epoch_total_acc:.4f} ALLAcc: {epoch_total_all_acc:.4f}')
            print(f'Micro| Precision: {total_f1score_dict["micro_precision"]}, Recall: {total_f1score_dict["micro_recall"]}, F1-score: {total_f1score_dict["micro_f1score"]}')
            print(f'Macro| Precision: {total_f1score_dict["macro_precision"]}, Recall: {total_f1score_dict["macro_recall"]}, F1-score: {total_f1score_dict["macro_f1score"]}')
            print('Abnormality' + '-'*40)
            print(f'Acc: {epoch_abnormality_acc:.4f}')
            print(f'Micro| Precision: {abnormality_f1score_dict["precision"]}, Recall: {abnormality_f1score_dict["recall"]}, F1-score: {abnormality_f1score_dict["f1score"]}')
            print('Label' + '-'*46)
            print(f'Acc: {epoch_label_acc:.4f} ALLAcc: {epoch_label_all_acc:.4f}')
            print(f'Micro| Precision: {label_f1score_dict["micro_precision"]}, Recall: {label_f1score_dict["micro_recall"]}, F1-score: {label_f1score_dict["micro_f1score"]}')
            print(f'Macro| Precision: {label_f1score_dict["macro_precision"]}, Recall: {label_f1score_dict["macro_recall"]}, F1-score: {total_f1score_dict["macro_f1score"]}')
            with open(train_result_txt_path, mode="a") as f:
                pprint(f'{phase}', stream=f)
                pprint('-' * 70, stream=f)
                pprint(f'Loss: {epoch_loss:.4f}', stream=f)
                pprint('Total' + '-'*46, stream=f)
                pprint(f'Acc: {epoch_total_acc:.4f} ALLAcc: {epoch_total_all_acc:.4f}', stream=f)
                pprint(f'Micro| Precision: {total_f1score_dict["micro_precision"]}, Recall: {total_f1score_dict["micro_recall"]}, F1-score: {total_f1score_dict["micro_f1score"]}', stream=f)
                pprint(f'Macro| Precision: {total_f1score_dict["macro_precision"]}, Recall: {total_f1score_dict["macro_recall"]}, F1-score: {total_f1score_dict["macro_f1score"]}', stream=f)
                pprint('Abnormality' + '-'*40, stream=f)
                pprint(f'Acc: {epoch_abnormality_acc:.4f}', stream=f)
                pprint(f'Micro| Precision: {abnormality_f1score_dict["precision"]}, Recall: {abnormality_f1score_dict["recall"]}, F1-score: {abnormality_f1score_dict["f1score"]}', stream=f)
                pprint('Label' + '-'*46, stream=f)
                pprint(f'Acc: {epoch_label_acc:.4f} ALLAcc: {epoch_label_all_acc:.4f}', stream=f)
                pprint(f'Micro| Precision: {label_f1score_dict["micro_precision"]}, Recall: {label_f1score_dict["micro_recall"]}, F1-score: {label_f1score_dict["micro_f1score"]}', stream=f)
                pprint(f'Macro| Precision: {label_f1score_dict["macro_precision"]}, Recall: {label_f1score_dict["macro_recall"]}, F1-score: {total_f1score_dict["macro_f1score"]}', stream=f)

            if phase == 'val' and total_f1score_dict['micro_f1score'] > best_f1score:
                best_f1score = total_f1score_dict['micro_f1score']
                best_model = model.state_dict()
                best_model_path = str(output_path / ('best_model-epoch' + str(epoch)))
                torch.save(best_model, best_model_path)
                best_model_wts = copy.deepcopy(best_model)
            if phase == 'val':
                val_f1score_history.append(total_f1score_dict['micro_f1score'])

                if epoch:
                    previous_model_path = str(output_path / ('model-epoch' + str(epoch-1)))
                    os.remove(previous_model_path)
                model_path = str(output_path / ('model-epoch' + str(epoch)))
                torch.save(model.state_dict(), model_path)

        print()
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best F1_Score Acc: {best_f1score:4f}')

    model.load_state_dict(best_model_wts)
    return model, val_f1score_history


def main(args, batch_size, epoch, dataset_name, model_name, seq=False, image_num_limit=60):
    print('loading dataset')
    file_path = Path(__file__)
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / dataset_name

    outputs_path = (file_path / '..' / '..' / 'outputs').resolve()
    output_dir = model_name + '-' + dataset_name
    output_path = outputs_path / output_dir
    output_dir_index = 0
    while os.path.isdir(output_path):
        output_dir_index += 1
        output_path = outputs_path / (output_dir + '-' + str(output_dir_index))
    os.makedirs(output_path, exist_ok=True)
    if not os.path.isdir(dataset_path):
        exit(1)

    shutil.copy(str(dataset_path / 'label_data.joblib'), str(output_path / 'label_data.joblib'))

    dataset_file = str(dataset_path / 'image_label.joblib')
    num_classes_path = str(dataset_path / 'num_classes.joblib')
    with open(num_classes_path, mode="rb") as f:
        num_classes = joblib.load(f)
    print('class num: ', num_classes)
    num_classes_path = str(output_path / 'num_classes.joblib')
    with open(num_classes_path, mode="wb") as f:
        joblib.dump(num_classes, f, compress=3)

    print('creating dataloaders')
    dataloaders_path = str(output_path / 'dataloaders.joblib')
    test_dataloader_path = str(output_path / 'test_dataloader.joblib')

    if seq:
        if 'resnet3d'in model_name:
            channel_first = True
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor()
            ])
            transform_3d = transforms.Compose([
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif model_name == 'lrcn':
            channel_first = False
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            transform_3d = None
        dataloaders = dataloader(dataset_file=dataset_file, dataloaders_path=dataloaders_path,
                                 test_dataloader_path=test_dataloader_path,
                                 transform=transform, transform_3d=transform_3d,
                                 batch_size=batch_size, image_num_limit=image_num_limit,
                                 seq=seq, channel_first=channel_first)
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # limit = 50000
        dataloaders = dataloader(dataset_file=dataset_file, dataloaders_path=dataloaders_path,
                                 test_dataloader_path=test_dataloader_path,
                                 transform=transform, batch_size=batch_size, seq=seq)
    print("creating model")
    model = None
    if model_name == 'mycnn':
        model = mycnn(num_classes)
    elif model_name == 'densenet_121':
        model = densenet_121(num_classes)
    elif model_name == 'densenet_121_expansion':
        model = densenet_121(num_classes, expansion=True)
    elif model_name == 'densenet_169':
        model = densenet_169(num_classes)
    elif model_name == 'densenet_169_expansion':
        model = densenet_169(num_classes, expansion=True)
    elif model_name == 'densenet_201':
        model = densenet_201(num_classes)
    elif model_name == 'densenet_201_expansion':
        model = densenet_201(num_classes, expansion=True)
    elif model_name == 'densenet_161':
        model = densenet_161(num_classes)
    elif model_name == 'densenet_161_expansion':
        model = densenet_161(num_classes, expansion=True)
    elif model_name == 'squeezenet':
        model = squeezenet(num_classes)
    elif model_name == 'resnet3d':
        model = resnet3d(num_classes)
    elif model_name == 'resnet3d_expansion':
        model = resnet3d(num_classes, expansion=True)
    elif model_name == 'resnet3d_maxpool':
        model = resnet3d(num_classes, expansion=False, maxpool=True)
    elif model_name == 'resnet3d_maxpool_expansion':
        model = resnet3d(num_classes, expansion=True, maxpool=True)
    elif model_name == 'lrcn':
        model = lrcn(num_classes, image_num_limit)
        image_num_limit_path = str(output_path / 'image_num_limit.joblib')
        with open(image_num_limit_path, mode="wb") as f:
            joblib.dump(image_num_limit, f, compress=3)
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

    print('description setting')
    train_description_path = str(output_path / 'train_description.txt')
    with open(train_description_path, mode="w") as f:
        pprint(str(args), stream=f)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', help='batch_size', type=int, default=50)
    parser.add_argument('-e', '--epoch', help='epoch', type=int, default=30)
    parser.add_argument('-d', '--dataset_name', help='dataset_name', type=str, required=True)
    parser.add_argument('-m', '--model_name', help='model_name', type=str, required=True)
    parser.add_argument('-s', '--seq', help='seq or not', action='store_true')
    parser.add_argument('-i', '--image_num_limit', help='image_num_limit', type=int, default=60)
    args = parser.parse_args()
    main(args, batch_size=args.batch_size, epoch=args.epoch,
         dataset_name=args.dataset_name, model_name=args.model_name,
         seq=args.seq, image_num_limit=args.image_num_limit)
