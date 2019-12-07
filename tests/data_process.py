import os
import glob
from pathlib import Path
import joblib
import pandas as pd
from tqdm import tqdm


def process0():
    file_path = Path(__file__)
    data_path = (file_path / '..' / '..' / 'data').resolve()
    kartes_path = str(data_path / '*' / '*.xlsx')
    images_path = str(data_path / '**' / '[0-1][0-9]' / '[0-3][0-9]' / '*' / '*')
    kartes = glob.glob(kartes_path)
    images = glob.iglob(images_path, recursive=True)

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

    image_path_list = list()
    labels_list = list()
    for image in tqdm(images):
        target_patient = int(image.split("/")[-2])
        target_data = all_karte[all_karte['primary'] == target_patient]
        if target_data['primary'].sum() == 0:
            continue
        image_path_list.append(image)
        labels_list.append(target_data['data_label'].to_list())

    data_dict = {
        'image_path': image_path_list,
        'labels': labels_list}
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / 'image-labels-ver0'
    os.makedirs(dataset_path, exist_ok=True)
    save_file_path = str(dataset_path / 'data.joblib')
    with open(save_file_path, mode="wb") as f:
        joblib.dump(data_dict, f, compress=3)


def process1():
    file_path = Path(__file__)
    data_path = (file_path / '..' / '..' / 'data').resolve()
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / 'labels'
    labels_file1 = str(dataset_path / 'labels1.joblib')
    images_path = str(data_path / '**' / '[0-1][0-9]' / '[0-3][0-9]' / '*' / '*')
    images = glob.iglob(images_path, recursive=True)

    with open(labels_file1, mode="rb") as f:
        primary_label_vector_dict = joblib.load(f)

    image_path_list = list()
    labels_list = list()
    for image in tqdm(images):
        target_patient = int(image.split("/")[-2])
        if target_patient not in primary_label_vector_dict.keys():
            continue
        target_data = primary_label_vector_dict[target_patient]
        image_path_list.append(image)
        labels_list.append(target_data)

    data_dict = {
        'image_path': image_path_list,
        'labels': labels_list}
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / 'image-labels-ver1'
    os.makedirs(dataset_path, exist_ok=True)
    save_file_path = str(dataset_path / 'data.joblib')
    print(save_file_path)
    with open(save_file_path, mode="wb") as f:
        joblib.dump(data_dict, f, compress=3)


def process2():
    file_path = Path(__file__)
    data_path = (file_path / '..' / '..' / 'data').resolve()
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / 'labels'
    labels_file2 = str(dataset_path / 'labels2.joblib')
    images_path = str(data_path / '**' / '[0-1][0-9]' / '[0-3][0-9]' / '*' / '*')
    images = glob.iglob(images_path, recursive=True)

    with open(labels_file2, mode="rb") as f:
        image_label_dict = joblib.load(f)

    image_path_list = list()
    labels_list = list()
    for image in tqdm(images):
        target_patient = int(image.split("/")[-2])
        if target_patient not in image_label_dict.keys():
            continue
        target_data = image_label_dict[target_patient]
        image_path_list.append(image)
        labels_list.append(target_data)

    data_dict = {
        'image_path': image_path_list,
        'labels': labels_list}
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / 'image-labels-ver2'
    os.makedirs(dataset_path, exist_ok=True)
    save_file_path = str(dataset_path / 'data.joblib')
    print(save_file_path)
    with open(save_file_path, mode="wb") as f:
        joblib.dump(data_dict, f, compress=3)


def process3():
    file_path = Path(__file__)
    data_path = (file_path / '..' / '..' / 'data').resolve()
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / 'labels'
    labels_file2 = str(dataset_path / 'labels3.joblib')
    images_path = str(data_path / '**' / '[0-1][0-9]' / '[0-3][0-9]' / '*' / '*')
    images = glob.iglob(images_path, recursive=True)

    with open(labels_file2, mode="rb") as f:
        image_label_dict = joblib.load(f)

    image_path_list = list()
    labels_list = list()
    for image in tqdm(images):
        target_patient = int(image.split("/")[-2])
        if target_patient not in image_label_dict.keys():
            continue
        target_data = image_label_dict[target_patient]
        image_path_list.append(image)
        labels_list.append(target_data)

    data_dict = {
        'image_path': image_path_list,
        'labels': labels_list}
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / 'image-labels-ver3'
    os.makedirs(dataset_path, exist_ok=True)
    save_file_path = str(dataset_path / 'data.joblib')
    print(save_file_path)
    with open(save_file_path, mode="wb") as f:
        joblib.dump(data_dict, f, compress=3)


def check01():
    file_path = Path(__file__)
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / 'labels'
    label_list_file1 = str(dataset_path / 'labels1list.joblib')
    with open(label_list_file1, mode="rb") as f:
        label_list = joblib.load(f)
    print(len(label_list))

    dataset_path = datasets_path / 'image-labels-ver1'
    save_file_path = str(dataset_path / 'data.joblib')
    with open(save_file_path, mode="rb") as f:
        data_dict = joblib.load(f)

    image_path_list = data_dict['image_path']
    labels_list = data_dict['labels']
    print(len(image_path_list))
    print(image_path_list[100])
    print(labels_list[100])

    dataset_path = datasets_path / 'image-labels-ver0'
    save_file_path = str(dataset_path / 'data.joblib')
    with open(save_file_path, mode="rb") as f:
        data_dict = joblib.load(f)

    image_path_list = data_dict['image_path']
    labels_list = data_dict['labels']
    print(len(image_path_list))
    print(image_path_list[100])
    print(labels_list[100])
    print(len(label_list))


def process4():
    file_path = Path(__file__)
    data_path = (file_path / '..' / '..' / 'data').resolve()
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / 'labels'
    labels_file2 = str(dataset_path / 'labels2.joblib')
    images_path = str(data_path / '**' / '[0-1][0-9]' / '[0-3][0-9]' / '*' / '*')
    images = glob.iglob(images_path, recursive=True)

    with open(labels_file2, mode="rb") as f:
        image_label_dict = joblib.load(f)

    image_path_list = list()
    labels_list = list()
    for image in tqdm(images):
        if os.stat(image).st_size == 0:
            continue
        target_patient = int(image.split("/")[-2])
        if target_patient not in image_label_dict.keys():
            continue
        target_data = image_label_dict[target_patient]
        image_path_list.append(image)
        labels_list.append(target_data)

    data_dict = {
        'image_path': image_path_list,
        'labels': labels_list}
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / 'image-labels-ver4'
    os.makedirs(dataset_path, exist_ok=True)
    save_file_path = str(dataset_path / 'data.joblib')
    print(save_file_path)
    with open(save_file_path, mode="wb") as f:
        joblib.dump(data_dict, f, compress=3)
    with open(save_file_path, mode="rb") as f:
        data_dict = joblib.load(f)

    image_path_list = data_dict['image_path']
    labels_list = data_dict['labels']
    print(len(image_path_list))
    print(image_path_list[100])
    print(labels_list[100])

    dataset_path = datasets_path / 'image-labels-ver2'
    save_file_path = str(dataset_path / 'data.joblib')
    with open(save_file_path, mode="rb") as f:
        data_dict = joblib.load(f)

    image_path_list = data_dict['image_path']
    labels_list = data_dict['labels']
    print(len(image_path_list))
    print(image_path_list[100])
    print(labels_list[100])


if __name__ == "__main__":
    print("process0")
    process0()
    print("process1")
    process1()
    check01()
    print("process2")
    process2()
    print("process3")
    process3()
    print("process4")
    process4()
