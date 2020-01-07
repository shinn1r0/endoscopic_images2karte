import argparse
import glob
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm


def pyplot_paper_init():
    # print(plt.rcParams.keys())
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.0


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def graph_plot(mode, results, result_name):
    pyplot_paper_init()
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    plt.gca().xaxis.get_major_formatter().set_useOffset(False)
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.locator_params(axis='y', nbins=20)
    plt.gca().yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)

    if mode == 'process':
        loss_list = results['epoch_loss_list']
        f1score_list = results['epoch_total_micro_f1score_list']
        label_f1score_list = results['epoch_label_micro_f1score_list']
        train_loss_list = np.array([loss.numpy() for i, loss in enumerate(loss_list) if i % 2 == 0])
        val_loss_list = np.array([loss.numpy() for i, loss in enumerate(loss_list) if i % 2 == 1])
        train_f1score_list = np.array([f1score for i, f1score in enumerate(f1score_list) if i % 2 == 0])
        val_f1score_list = np.array([f1score for i, f1score in enumerate(f1score_list) if i % 2 == 1])
        train_label_f1score_list = \
            np.array([label_f1score for i, label_f1score in enumerate(label_f1score_list) if i % 2 == 0])
        val_label_f1score_list = \
            np.array([label_f1score for i, label_f1score in enumerate(label_f1score_list) if i % 2 == 1])

        # fig, ax1 = plt.subplots(figsize=(3.14, 3.14))
        fig, ax1 = plt.subplots()
        ax1.plot(train_loss_list, label='Train Loss', color='black', linestyle='--')
        ax1.plot(val_loss_list, label='Validation Loss', color='red', linestyle='--')

        ax2 = ax1.twinx()
        ax2.plot(train_f1score_list * 100, label='Train F1-Score', color='black', linestyle='-')
        ax2.plot(val_f1score_list * 100, label='Validation F1-Score', color='red', linestyle='-')

        ax1.set_xlim(0, 30)
        ax1.set_xlabel('Number of Learning    [epoch]')

        ax1.set_ylim(0, 0.30)
        ax1.set_yticks(np.arange(0, 0.31, 0.05))
        ax1.set_ylabel('Loss')

        ax2.set_ylim(50, 100)
        ax2.set_yticks(np.arange(50, 101, 10))
        ax2.set_ylabel('F1-Score    [%]')

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='lower right')
    elif mode == 'threshold':
        precision_list = list()
        recall_list = list()
        f1score_list = list()
        for result in results:
            precision_list.append(result['label_micro_precision'])
            recall_list.append(result['label_micro_recall'])
            f1score_list.append(result['label_micro_f1score'])
        precision_list = np.array(precision_list)
        recall_list = np.array(recall_list)
        f1score_list = np.array(f1score_list)

        # fig, ax1 = plt.subplots(figsize=(3.14, 3.14))
        fig, ax1 = plt.subplots()
        threshold = np.arange(0.10, 0.91, 0.10)
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.30))
        make_patch_spines_invisible(ax3)
        ax3.spines['right'].set_visible(True)
        ax1.set_xlim(0, 1)
        ax1.set_xticks(np.arange(0.10, 0.91, 0.20))
        ax1.set_xlabel('threshold')

        ax1.plot(threshold, precision_list * 100, label='Precision', color='blue', linestyle='-')
        ax1.set_yticks(np.arange(0, 101, 20))
        ax1.set_ylabel('Precision (%)')
        ax2.plot(threshold, recall_list * 100, label='Recall', color='red', linestyle='-')
        ax2.set_yticks(np.arange(0, 101, 20))
        ax2.set_ylabel('Recall (%)')
        ax3.plot(threshold, f1score_list * 100, label='F1-Score', color='black', linestyle='-')
        ax3.set_yticks(np.arange(0, 101, 20))
        ax3.set_ylabel('F1-Score (%)')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        h3, l3 = ax3.get_legend_handles_labels()
        ax1.legend(h1+h2+h3, l1+l2+l3, loc='lower left')

    plt.tight_layout()
    plt.savefig('./fig/' + result_name + '.pdf', transparent=True)
    plt.savefig('./fig/' + result_name + '.png', transparent=True, dpi=300)


def main(result_dir, mode):
    print('loading result')
    file_path = Path(__file__)
    outputs_path = (file_path / '..' / '..' / 'outputs').resolve()
    if mode == 'process':
        output_path = outputs_path / result_dir
        train_result_path = glob.glob(str(output_path / 'train_result.joblib'))[0]
        with open(train_result_path, 'rb') as f:
            train_result_dict = joblib.load(f)
        graph_plot(mode, train_result_dict, result_dir+mode)
    elif mode == 'score':
        output_path = outputs_path / result_dir
        train_result_path = glob.glob(str(output_path / 'train_result.joblib'))[0]
        with open(train_result_path, 'rb') as f:
            train_result_dict = joblib.load(f)
        f1score_list = np.array([x for i, x in enumerate(train_result_dict['epoch_total_micro_f1score_list']) if i % 2 == 1])
        max_index = np.argmax(f1score_list)
        acc_list = np.array([x for i, x in enumerate(train_result_dict['epoch_total_acc_list']) if i % 2 == 1])
        all_acc_list = np.array([x for i, x in enumerate(train_result_dict['epoch_total_all_acc_list']) if i % 2 == 1])
        precision_list = np.array([x for i, x in enumerate(train_result_dict['epoch_total_micro_precision_list']) if i % 2 == 1])
        recall_list = np.array([x for i, x in enumerate(train_result_dict['epoch_total_micro_recall_list']) if i % 2 == 1])

        lesion_acc_list = np.array([x for i, x in enumerate(train_result_dict['epoch_abnormality_acc_list']) if i % 2 == 1])
        lesion_precision_list = np.array([x for i, x in enumerate(train_result_dict['epoch_abnormality_precision_list']) if i % 2 == 1])
        lesion_recall_list = np.array([x for i, x in enumerate(train_result_dict['epoch_abnormality_recall_list']) if i % 2 == 1])
        lesion_f1score_list = np.array([x for i, x in enumerate(train_result_dict['epoch_abnormality_f1score_list']) if i % 2 == 1])

        label_acc_list = np.array([x for i, x in enumerate(train_result_dict['epoch_label_acc_list']) if i % 2 == 1])
        label_all_acc_list = np.array([x for i, x in enumerate(train_result_dict['epoch_label_all_acc_list']) if i % 2 == 1])
        label_precision_list = np.array([x for i, x in enumerate(train_result_dict['epoch_label_micro_precision_list']) if i % 2 == 1])
        label_recall_list = np.array([x for i, x in enumerate(train_result_dict['epoch_label_micro_recall_list']) if i % 2 == 1])
        label_f1score_list = np.array([x for i, x in enumerate(train_result_dict['epoch_label_micro_f1score_list']) if i % 2 == 1])
        print(result_dir)
        print('-' * 100)
        print('acc')
        print(acc_list[max_index])
        print('all_acc')
        print(all_acc_list[max_index])
        print('precision')
        print(precision_list[max_index])
        print('recall')
        print(recall_list[max_index])
        print('f1score')
        print(f1score_list[max_index])
        print('lesion_acc')
        print(lesion_acc_list[max_index])
        print('lesion_precision')
        print(lesion_precision_list[max_index])
        print('lesion_recall')
        print(lesion_recall_list[max_index])
        print('lesion_f1score')
        print(lesion_f1score_list[max_index])
        print('label_acc')
        print(label_acc_list[max_index])
        print('label_all_acc')
        print(label_all_acc_list[max_index])
        print('label_precision')
        print(label_precision_list[max_index])
        print('label_recall')
        print(label_recall_list[max_index])
        print('label_f1score')
        print(label_f1score_list[max_index])
    elif mode == 'threshold':
        output_path = outputs_path / ('test-' + result_dir + '_ith-none_lth-*')
        test_result_paths = glob.glob(str(output_path / 'test_result.joblib'))
        test_result_paths = [x for x in test_result_paths if 0.0 < float(x.split('lth-')[1].split('/')[0])]
        test_result_paths = sorted(test_result_paths, key=lambda x: float(x.split('lth-')[1].split('/')[0]))
        test_result_dict_list = list()
        for test_result_path in tqdm(test_result_paths):
            with open(test_result_path, 'rb') as f:
                test_result_dict = joblib.load(f)
                test_result_dict_list.append(test_result_dict)
        graph_plot(mode, test_result_dict_list, result_dir+mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--result_dir', help='result_dir', type=str, required=True)
    parser.add_argument('-m', '--mode', help='mode', type=str, required=True)
    args = parser.parse_args()
    main(result_dir=args.result_dir, mode=args.mode)
