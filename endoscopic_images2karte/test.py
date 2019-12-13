import argparse
import glob
import os
import time
from pathlib import Path
from pprint import pprint

import joblib
import numpy as np
from sklearn import metrics
import torch
from tqdm import tqdm

from models import mycnn, densenet_121, densenet_169, densenet_201, densenet_161, resnet3d, lrcn
from utils import calc_f1score


def test_model(model, test_dataloader, num_classes, label_data, seq, flag_it, image_threshold, label_threshold, verbose,
               device, output_path, test_result_txt_path, test_result_dict_path):
    """
    Test Model

    Args:
        model (torch.nn.modules.module.Module):
        test_dataloader (str, torch.util.data.Dataloader):
        num_classes (int):
        device (torch.device):
        output_path (str):
        test_result_txt_path (str):
        test_result_dict_path (str):

    Returns:
        None
    """
    since = time.time()
    test_result_dict = {
        'label_list': list(),
        'preds_t_list_list': list(),
        'preds_list': list(),
    }

    model.eval()
    dataset_length = len(test_dataloader.dataset)
    all_label_list = np.array(label_data['all_label_list'])

    running_mcm = np.zeros([num_classes, 2, 2], dtype=np.int)
    running_total_corrects = torch.tensor(0.0)
    running_total_all_corrects = torch.tensor(0.0)
    running_abnormality_corrects = torch.tensor(0.0)
    running_label_corrects = torch.tensor(0.0)
    running_label_all_corrects = torch.tensor(0.0)
    for inputs, labels in tqdm(test_dataloader):
        labels = labels.to(device)
        test_result_dict['label_list'].append(labels)
        if seq:
            labels_list = labels.unbind(dim=0)
            inputs = inputs.to(device)
            with torch.set_grad_enabled(False):
                preds = torch.sigmoid(model(inputs))
            preds_list = (preds > label_threshold).float().unbind(dim=0)
        else:
            if flag_it:
                preds = list()
            else:
                outputs = list()
            for inputs_t in inputs:
                inputs_t = inputs_t.to(device)
                with torch.set_grad_enabled(False):
                    outputs_t = torch.sigmoid(model(inputs_t))
                    if flag_it:
                        preds_t = (outputs_t > image_threshold).float()
                        preds.append(preds_t)
                    else:
                        outputs.append(outputs_t)
            if flag_it:
                test_result_dict['preds_t_list_list'].append(preds)
                preds = torch.stack(preds, dim=0).mean(dim=0)
            else:
                test_result_dict['preds_t_list_list'].append(outputs)
                preds = torch.stack(outputs, dim=0).mean(dim=0)

            preds = (preds > label_threshold).float()

        test_result_dict['preds_list'].append(preds)

        if seq:
            for labels, preds in zip(labels_list, preds_list):
                labels = labels.unsqueeze(dim=0)
                preds = preds.unsqueeze(dim=0)
                total_corrects = torch.eq(preds, labels.data).float().sum(dim=1) / num_classes
                abnormality_corrects = torch.eq(preds[:, 0], labels.data[:, 0]).int()
                label_corrects = torch.eq(preds[:, 1:], labels.data[:, 1:]).float().sum(dim=1) / (num_classes - 1)

                running_mcm += metrics.multilabel_confusion_matrix(labels.data.cpu(), preds.cpu())
                running_total_corrects += total_corrects.sum(dim=0)
                running_total_all_corrects += total_corrects.int().sum(dim=0)
                running_abnormality_corrects += abnormality_corrects.sum(dim=0)
                running_label_corrects += label_corrects.sum(dim=0)
                running_label_all_corrects += label_corrects.int().sum(dim=0)

                labels = labels.squeeze().cpu().numpy()
                preds = preds.squeeze().cpu().numpy()
                with open(test_result_txt_path, mode="a") as f:
                    pprint('Label' + '-'*80, stream=f)
                    pprint(labels[0], stream=f)
                    pprint(all_label_list[labels[1:] == 1.0], stream=f)
                    if verbose:
                        pprint(labels, stream=f)
                    pprint('Pred' + '-'*81, stream=f)
                    pprint(preds[0], stream=f)
                    pprint(all_label_list[preds[1:] == 1.0], stream=f)

        else:
            total_corrects = torch.eq(preds, labels.data).float().sum(dim=1) / num_classes
            abnormality_corrects = torch.eq(preds[:, 0], labels.data[:, 0]).int()
            label_corrects = torch.eq(preds[:, 1:], labels.data[:, 1:]).float().sum(dim=1) / (num_classes - 1)

            running_mcm += metrics.multilabel_confusion_matrix(labels.data.cpu(), preds.cpu())
            running_total_corrects += total_corrects.sum(dim=0)
            running_total_all_corrects += total_corrects.int().sum(dim=0)
            running_abnormality_corrects += abnormality_corrects.sum(dim=0)
            running_label_corrects += label_corrects.sum(dim=0)
            running_label_all_corrects += label_corrects.int().sum(dim=0)

            labels = labels.squeeze().cpu().numpy()
            preds = preds.squeeze().cpu().numpy()
            with open(test_result_txt_path, mode="a") as f:
                pprint('Label' + '-'*80, stream=f)
                pprint(labels[0], stream=f)
                pprint(all_label_list[labels[1:] == 1.0], stream=f)
                if verbose:
                    pprint(labels, stream=f)
                pprint('Pred' + '-'*81, stream=f)
                pprint(preds[0], stream=f)
                pprint(all_label_list[preds[1:] == 1.0], stream=f)
                if verbose:
                    pprint(preds, stream=f)
                    pprint(test_result_dict['preds_t_list_list'][-1], stream=f)

    total_f1score_dict = calc_f1score(running_mcm, multi=True)
    abnormality_f1score_dict = calc_f1score(running_mcm[0], multi=False)
    label_f1score_dict = calc_f1score(running_mcm[1:], multi=True)
    total_acc = (running_total_corrects.double() / dataset_length).item()
    total_all_acc = (running_total_all_corrects.double() / dataset_length).item()
    abnormality_acc = (running_abnormality_corrects.double() / dataset_length).item()
    label_acc = (running_label_corrects.double() / dataset_length).item()
    label_all_acc = (running_label_all_corrects.double() / dataset_length).item()

    test_result_dict['total_micro_precision'] = total_f1score_dict['micro_precision']
    test_result_dict['total_micro_recall'] = total_f1score_dict['micro_recall']
    test_result_dict['total_micro_f1score'] = total_f1score_dict['micro_f1score']
    test_result_dict['total_macro_precision'] = total_f1score_dict['macro_precision']
    test_result_dict['total_macro_recall'] = total_f1score_dict['macro_recall']
    test_result_dict['total_macro_f1score'] = total_f1score_dict['macro_f1score']
    test_result_dict['total_acc'] = total_acc
    test_result_dict['total_all_acc'] = total_all_acc
    test_result_dict['abnormality_precision'] = abnormality_f1score_dict['precision']
    test_result_dict['abnormality_recall'] = abnormality_f1score_dict['recall']
    test_result_dict['abnormality_f1score'] = abnormality_f1score_dict['f1score']
    test_result_dict['abnormality_acc'] = abnormality_acc
    test_result_dict['label_micro_precision'] = label_f1score_dict['micro_precision']
    test_result_dict['label_micro_recall'] = label_f1score_dict['micro_recall']
    test_result_dict['label_micro_f1score'] = label_f1score_dict['micro_f1score']
    test_result_dict['label_macro_precision'] = label_f1score_dict['macro_precision']
    test_result_dict['label_macro_recall'] = label_f1score_dict['macro_recall']
    test_result_dict['label_macro_f1score'] = label_f1score_dict['macro_f1score']
    test_result_dict['label_acc'] = label_acc
    test_result_dict['label_all_acc'] = label_all_acc
    with open(test_result_dict_path, mode="wb") as f:
        joblib.dump(test_result_dict, f, compress=3)

    print('Total' + '-'*46)
    print(f'Acc: {total_acc:.4f} ALLAcc: {total_all_acc:.4f}')
    print(f'Micro| Precision: {total_f1score_dict["micro_precision"]}, Recall: {total_f1score_dict["micro_recall"]}, F1-score: {total_f1score_dict["micro_f1score"]}')
    print(f'Macro| Precision: {total_f1score_dict["macro_precision"]}, Recall: {total_f1score_dict["macro_recall"]}, F1-score: {total_f1score_dict["macro_f1score"]}')
    print('Abnormality' + '-'*40)
    print(f'Acc: {abnormality_acc:.4f}')
    print(f'Micro| Precision: {abnormality_f1score_dict["precision"]}, Recall: {abnormality_f1score_dict["recall"]}, F1-score: {abnormality_f1score_dict["f1score"]}')
    print('Label' + '-'*46)
    print(f'Acc: {label_acc:.4f} ALLAcc: {label_all_acc:.4f}')
    print(f'Micro| Precision: {label_f1score_dict["micro_precision"]}, Recall: {label_f1score_dict["micro_recall"]}, F1-score: {label_f1score_dict["micro_f1score"]}')
    print(f'Macro| Precision: {label_f1score_dict["macro_precision"]}, Recall: {label_f1score_dict["macro_recall"]}, F1-score: {total_f1score_dict["macro_f1score"]}')
    with open(test_result_txt_path, mode="a") as f:
        pprint('Total' + '-'*46, stream=f)
        pprint(f'Acc: {total_acc:.4f} ALLAcc: {total_all_acc:.4f}', stream=f)
        pprint(f'Micro| Precision: {total_f1score_dict["micro_precision"]}, Recall: {total_f1score_dict["micro_recall"]}, F1-score: {total_f1score_dict["micro_f1score"]}', stream=f)
        pprint(f'Macro| Precision: {total_f1score_dict["macro_precision"]}, Recall: {total_f1score_dict["macro_recall"]}, F1-score: {total_f1score_dict["macro_f1score"]}', stream=f)
        pprint('Abnormality' + '-'*40, stream=f)
        pprint(f'Acc: {abnormality_acc:.4f}', stream=f)
        pprint(f'Micro| Precision: {abnormality_f1score_dict["precision"]}, Recall: {abnormality_f1score_dict["recall"]}, F1-score: {abnormality_f1score_dict["f1score"]}', stream=f)
        pprint('Label' + '-'*46, stream=f)
        pprint(f'Acc: {label_acc:.4f} ALLAcc: {label_all_acc:.4f}', stream=f)
        pprint(f'Micro| Precision: {label_f1score_dict["micro_precision"]}, Recall: {label_f1score_dict["micro_recall"]}, F1-score: {label_f1score_dict["micro_f1score"]}', stream=f)
        pprint(f'Macro| Precision: {label_f1score_dict["macro_precision"]}, Recall: {label_f1score_dict["macro_recall"]}, F1-score: {total_f1score_dict["macro_f1score"]}', stream=f)

    print()
    time_elapsed = time.time() - since
    print(f'Testing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')


def main(args, result_dir, seq, flag_it, image_threshold, label_threshold, verbose, gpu):
    print('loading best model and test_dataloader')
    file_path = Path(__file__)
    outputs_path = (file_path / '..' / '..' / 'outputs').resolve()
    output_path = outputs_path / result_dir
    best_model_paths = glob.glob(str(output_path / 'best_model*'))
    best_model_path = sorted(best_model_paths, key=lambda x: int(x.split('epoch')[1]))[-1]
    num_classes_path = str(output_path / 'num_classes.joblib')
    label_data_path = str(output_path / 'label_data.joblib')
    with open(label_data_path, mode="rb") as f:
        label_data = joblib.load(f)
    with open(num_classes_path, mode="rb") as f:
        num_classes = joblib.load(f)
    if 'mycnn' in result_dir:
        model = mycnn(num_classes)
        model.load_state_dict(torch.load(best_model_path))
    elif 'densenet121_e' in result_dir:
        model = densenet_121(num_classes, expansion=True)
        model.load_state_dict(torch.load(best_model_path))
    elif 'densenet121' in result_dir:
        model = densenet_121(num_classes)
        model.load_state_dict(torch.load(best_model_path))
    elif 'densenet161_e' in result_dir:
        model = densenet_161(num_classes, expansion=True)
        model.load_state_dict(torch.load(best_model_path))
    elif 'densenet161' in result_dir:
        model = densenet_161(num_classes)
        model.load_state_dict(torch.load(best_model_path))
    elif 'resnet3d_e_m' in result_dir:
        model = resnet3d(num_classes, expansion=True, maxpool=True)
        model.load_state_dict(torch.load(best_model_path))
    elif 'resnet3d_m' in result_dir:
        model = resnet3d(num_classes, expansion=False, maxpool=True)
        model.load_state_dict(torch.load(best_model_path))
    elif 'resnet3d_e' in result_dir:
        model = resnet3d(num_classes, expansion=True)
        model.load_state_dict(torch.load(best_model_path))
    elif 'resnet3d' in result_dir:
        model = resnet3d(num_classes)
        model.load_state_dict(torch.load(best_model_path))
    elif 'lrcn' in result_dir:
        image_num_limit_path = str(output_path / 'image_num_limit.joblib')
        with open(image_num_limit_path, mode="rb") as f:
            image_num_limit = joblib.load(f)
        model = lrcn(num_classes, image_num_limit)
        model.load_state_dict(torch.load(best_model_path))
    else:
        exit(1)
    test_dataloader_path = str(output_path / 'test_dataloader.joblib')
    with open(test_dataloader_path, mode="rb") as f:
        test_dataloader = joblib.load(f)

    print("cuda settings")
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    print('start testing')
    if flag_it:
        test_output_path = outputs_path / ('test-' + result_dir + '_ith-' + str(image_threshold) + '_lth-' + str(label_threshold))
    else:
        test_output_path = outputs_path / ('test-' + result_dir + '_ith-' + 'none' + '_lth-' + str(label_threshold))
    output_dir_index = 0
    while os.path.isdir(test_output_path):
        output_dir_index += 1
        if flag_it:
            test_output_path = outputs_path / ('test-' + result_dir + '_ith-' + str(image_threshold) + '_lth-' + str(label_threshold) + '-' + str(output_dir_index))
        else:
            test_output_path = outputs_path / ('test-' + result_dir + '_ith-' + 'none' + '_lth-' + str(label_threshold) + '-' + str(output_dir_index))
    os.makedirs(test_output_path, exist_ok=True)
    test_result_txt_path = str(test_output_path / 'test_result.txt')
    test_result_dict_path = str(test_output_path / 'test_result.joblib')
    test_model(model=model, test_dataloader=test_dataloader, num_classes=num_classes, label_data=label_data, seq=seq,
               flag_it=flag_it, image_threshold=image_threshold, label_threshold=label_threshold,
               verbose=verbose, device=device, output_path=output_path,
               test_result_txt_path=test_result_txt_path, test_result_dict_path=test_result_dict_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--result_dir', help='result_dir', type=str, required=True)
    parser.add_argument('-s', '--seq', help='seq or not', action='store_true')
    parser.add_argument('-i', '--flag_image_threshold', help='using image_threshold_or_not', action='store_true')
    parser.add_argument('--image_threshold', help='image_threshold', type=float, required=True)
    parser.add_argument('--label_threshold', help='label_threshold', type=float, required=True)
    parser.add_argument('-v', '--verbose', help='verbose or not', action='store_true')
    parser.add_argument('-g', '--gpu', help='gpu or cpu', action='store_true')
    args = parser.parse_args()
    main(args, result_dir=args.result_dir, seq=args.seq, flag_it=args.flag_image_threshold,
         image_threshold=args.image_threshold, label_threshold=args.label_threshold, verbose=args.verbose, gpu=args.gpu)
