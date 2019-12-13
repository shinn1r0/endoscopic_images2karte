import argparse
import copy
import os
import time
from pathlib import Path
from pprint import pprint
import glob

import joblib
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models import mycnn, densenet_121, densenet_169, densenet_201, densenet_161, resnet3d, lrcn
from utils import calc_f1score


def retrain_model(model, dataloaders, criterion, optimizer, num_classes, num_epochs, last_epoch, best_epoch,
                  device, output_path, train_result_txt_path, train_result_dict_path):
    """
    Retrain Model

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
    with open(train_result_dict_path, mode="rb") as f:
        train_result_dict = joblib.load(f)
    val_f1score_history = list()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1score = train_result_dict['epoch_total_micro_f1score_list'][best_epoch]

    for epoch in range(last_epoch+1, last_epoch+1+num_epochs):
        print(f'Epoch {epoch}/{last_epoch+num_epochs}')
        print('=' * 80)
        with open(train_result_txt_path, mode="a") as f:
            pprint(f'Epoch {epoch}/{last_epoch+num_epochs}', stream=f)
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


def main(args, epoch, result_dir, seq):
    print('loading last epoch model and dataloaders')
    file_path = Path(__file__)
    outputs_path = (file_path / '..' / '..' / 'outputs').resolve()
    output_path = outputs_path / result_dir
    last_epoch_model_path = glob.glob(str(output_path / 'model-epoch*'))[0]
    last_epoch = int(last_epoch_model_path.split('epoch')[1])
    best_model_paths = glob.glob(str(output_path / 'best_model*'))
    best_model_path = sorted(best_model_paths, key=lambda x: int(x.split('epoch')[1]))[-1]
    best_epoch = int(best_model_path.split('epoch')[1])
    num_classes_path = str(output_path / 'num_classes.joblib')
    with open(num_classes_path, mode="rb") as f:
        num_classes = joblib.load(f)
    if 'mycnn' in result_dir:
        model = mycnn(num_classes)
        model.load_state_dict(torch.load(last_epoch_model_path))
    elif 'densenet121_e' in result_dir:
        model = densenet_121(num_classes, expansion=True)
        model.load_state_dict(torch.load(last_epoch_model_path))
    elif 'densenet121' in result_dir:
        model = densenet_121(num_classes)
        model.load_state_dict(torch.load(last_epoch_model_path))
    elif 'densenet161_e' in result_dir:
        model = densenet_161(num_classes, expansion=True)
        model.load_state_dict(torch.load(last_epoch_model_path))
    elif 'densenet161' in result_dir:
        model = densenet_161(num_classes)
        model.load_state_dict(torch.load(last_epoch_model_path))
    elif 'resnet3d_e_m' in result_dir:
        model = resnet3d(num_classes, expansion=True, maxpool=True)
        model.load_state_dict(torch.load(last_epoch_model_path))
    elif 'resnet3d_m' in result_dir:
        model = resnet3d(num_classes, expansion=False, maxpool=True)
        model.load_state_dict(torch.load(last_epoch_model_path))
    elif 'resnet3d_e' in result_dir:
        model = resnet3d(num_classes, expansion=True)
        model.load_state_dict(torch.load(last_epoch_model_path))
    elif 'resnet3d' in result_dir:
        model = resnet3d(num_classes)
        model.load_state_dict(torch.load(last_epoch_model_path))
    elif 'lrcn' in result_dir:
        image_num_limit_path = str(output_path / 'image_num_limit.joblib')
        with open(image_num_limit_path, mode="rb") as f:
            image_num_limit = joblib.load(f)
        model = lrcn(num_classes, image_num_limit)
        model.load_state_dict(torch.load(last_epoch_model_path))
    else:
        exit(1)
    dataloaders_path = str(output_path / 'dataloaders.joblib')
    with open(dataloaders_path, mode="rb") as f:
        dataloaders = joblib.load(f)

    print("cuda settings")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print('optimizer and criterion setting')
    optimizer = optim.Adam(model.parameters())
    weight = torch.ones([num_classes]).to(device)
    criterion = nn.BCEWithLogitsLoss(weight=weight)

    print('start retraining')
    train_result_txt_path = str(output_path / 'train_result.txt')
    train_result_dict_path = str(output_path / 'train_result.joblib')
    model, hist = retrain_model(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer,
                                num_classes=num_classes, num_epochs=epoch, last_epoch=last_epoch, best_epoch=best_epoch,
                                device=device, output_path=output_path,
                                train_result_txt_path=train_result_txt_path,
                                train_result_dict_path=train_result_dict_path)
    print(hist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', help='additional epoch', type=int, default=10)
    parser.add_argument('-r', '--result_dir', help='result_dir', type=str, required=True)
    parser.add_argument('-s', '--seq', help='seq or not', action='store_true')
    args = parser.parse_args()
    main(args, epoch=args.epoch, result_dir=args.result_dir, seq=args.seq)
