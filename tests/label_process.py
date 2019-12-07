import os
import glob
from pathlib import Path
import joblib
import pandas as pd
import torch
from tqdm import tqdm


def process0():
    file_path = Path(__file__)
    data_path = (file_path / '..' / '..' / 'data').resolve()
    kartes_path = str(data_path / '*' / '*.xlsx')
    kartes = glob.glob(kartes_path)
    karte_list = list()
    for karte in tqdm(kartes):
        karte_data = pd.read_excel(karte)
        if len(karte_data.columns) >= 10:
            karte_data = karte_data.rename(columns={
                '検査プライマリ': 'primary',
                '質的診断': 'data_label'})
            karte_list.append(karte_data)
    all_karte = pd.concat(karte_list, sort=False)
    all_karte = all_karte.dropna(subset=['primary']).astype({'primary': int})
    all_karte = all_karte.drop_duplicates(keep='first')
    print(all_karte.info())

    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / 'kartes'
    os.makedirs(dataset_path, exist_ok=True)
    all_karte_file = str(dataset_path / 'all_karte.joblib')
    with open(all_karte_file, mode="wb") as f:
        joblib.dump(all_karte, f, compress=3)
    with open(all_karte_file, mode="rb") as f:
        all_karte = joblib.load(f)
    print(all_karte.info())

    all_karte = all_karte[['data_label', 'primary']]
    primary = None
    labels = None
    primary_labels_dict = dict()

    for karte in tqdm(all_karte.itertuples()):
        if primary is None or primary != karte.primary:
            if primary is not None:
                if primary in primary_labels_dict.keys():
                    primary_labels_dict[primary].extend(labels)
                else:
                    primary_labels_dict[primary] = labels
            primary = karte.primary
            labels = list()
        labels.append(karte.data_label)
    else:
        if primary in primary_labels_dict.keys():
            primary_labels_dict[primary].extend(labels)
        else:
            primary_labels_dict[primary] = labels

    print(len(primary_labels_dict.keys()))
    dataset_path = datasets_path / 'labels'
    os.makedirs(dataset_path, exist_ok=True)
    labels_file0 = str(dataset_path / 'labels0.joblib')
    with open(labels_file0, mode="wb") as f:
        joblib.dump(primary_labels_dict, f, compress=3)


def process1():
    file_path = Path(__file__)
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / 'labels'
    labels_file0 = str(dataset_path / 'labels0.joblib')
    with open(labels_file0, mode="rb") as f:
        primary_labels_dict = joblib.load(f)
    print(primary_labels_dict[501519])

    label_list = list()
    primary_label_vector_dict = dict()
    for primary, labels in tqdm(primary_labels_dict.items()):
        label_vector = list()
        for label in labels:
            if label in label_list:
                label_vector.append(label_list.index(label))
            else:
                label_vector.append(len(label_list))
                label_list.append(label)
        primary_label_vector_dict[primary] = sorted(list(set(label_vector)))
    labels_file1 = str(dataset_path / 'labels1.joblib')
    label_list_file1 = str(dataset_path / 'labels1list.joblib')
    with open(labels_file1, mode="wb") as f:
        joblib.dump(primary_label_vector_dict, f, compress=3)
    with open(label_list_file1, mode="wb") as f:
        joblib.dump(label_list, f, compress=3)

    with open(labels_file1, mode="rb") as f:
        primary_label_vector_dict = joblib.load(f)
    with open(label_list_file1, mode="rb") as f:
        label_list = joblib.load(f)

    count = 0
    for primary, label_vector in primary_label_vector_dict.items():
        print(primary)
        print(label_vector)
        count += 1
        if count > 100:
            break
    print(label_list[10])


def process2():
    file_path = Path(__file__)
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / 'labels'
    labels_file1 = str(dataset_path / 'labels1.joblib')
    label_list_file1 = str(dataset_path / 'labels1list.joblib')
    with open(labels_file1, mode="rb") as f:
        primary_label_vector_dict = joblib.load(f)
    with open(label_list_file1, mode="rb") as f:
        label_list = joblib.load(f)
    image_label_dict = dict()
    for primary, labels in tqdm(primary_label_vector_dict.items()):
        labels = torch.zeros(len(label_list)).scatter(0, torch.tensor(labels), 1.)
        image_label_dict[primary] = labels

    labels_file2 = str(dataset_path / 'labels2.joblib')
    label_list_file2 = str(dataset_path / 'labels2list.joblib')
    with open(labels_file2, mode="wb") as f:
        joblib.dump(image_label_dict, f, compress=3)
    with open(label_list_file2, mode="wb") as f:
        joblib.dump(label_list, f, compress=3)
    with open(labels_file2, mode="rb") as f:
        image_label_dict = joblib.load(f)
    with open(label_list_file2, mode="rb") as f:
        label_list = joblib.load(f)
    count = 0
    for primary, label in image_label_dict.items():
        print(primary)
        print(label)
        count += 1
        if count > 100:
            break


def process3():
    file_path = Path(__file__)
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / 'labels'
    labels_file1 = str(dataset_path / 'labels1.joblib')
    label_list_file1 = str(dataset_path / 'labels1list.joblib')
    with open(labels_file1, mode="rb") as f:
        primary_label_vector_dict = joblib.load(f)
    with open(label_list_file1, mode="rb") as f:
        label_list = joblib.load(f)
    image_label_dict = dict()
    for primary, labels in tqdm(primary_label_vector_dict.items()):
        labels = torch.tensor(labels)
        labels = labels.unsqueeze(0)
        labels = torch.zeros(labels.size(0), len(label_list)).scatter(1, labels, 1.)
        image_label_dict[primary] = labels

    labels_file3 = str(dataset_path / 'labels3.joblib')
    label_list_file3 = str(dataset_path / 'labels3list.joblib')
    with open(labels_file3, mode="wb") as f:
        joblib.dump(image_label_dict, f, compress=3)
    with open(label_list_file3, mode="wb") as f:
        joblib.dump(label_list, f, compress=3)
    with open(labels_file3, mode="rb") as f:
        image_label_dict = joblib.load(f)
    with open(label_list_file3, mode="rb") as f:
        label_list = joblib.load(f)
    count = 0
    for primary, label in image_label_dict.items():
        print(primary)
        print(label)
        print(label.size())
        count += 1
        if count > 100:
            break


def check0():
    file_path = Path(__file__)
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / 'labels'
    labels_file0 = str(dataset_path / 'labels0.joblib')
    with open(labels_file0, mode="rb") as f:
        primary_labels_dict = joblib.load(f)
    sum = 0
    for labels in tqdm(primary_labels_dict.values()):
        sum += len(labels)
    print(sum)
    print(len(primary_labels_dict.keys()))


if __name__ == "__main__":
    print("process0")
    process0()
    print("process1")
    process1()
    print("process2")
    process2()
    check0()
    print("process3")
    process3()
