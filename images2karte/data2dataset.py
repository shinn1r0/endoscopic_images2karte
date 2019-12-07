import glob
import itertools
import os
from pathlib import Path
from pprint import pprint

import joblib
import pandas as pd
from tqdm import tqdm

from vector_process import create_label_lib, choose_cut_line
from vector_process import vectorize, onehot_vectorize, nested_onehot_vectorize


def save_concat_karte(kartes_path, all_karte_path):
    kartes = glob.glob(kartes_path)
    karte_list = list()
    for karte in kartes:
        karte_data = pd.read_excel(karte)
        if len(karte_data.columns) >= 10:
            karte_data = karte_data.rename(columns={
                '検査プライマリ': 'primary',
                '質的診断': 'data_label'})
            karte_list.append(karte_data)
    all_karte = pd.concat(karte_list, sort=False)
    all_karte = all_karte.astype({'primary': int})
    all_karte = all_karte.drop_duplicates(keep='first')
    print(all_karte.info())

    with open(all_karte_path, mode="wb") as f:
        joblib.dump(all_karte, f, compress=3)


def save_extracted_karte(all_karte_path, primary_images_dict, extracted_karte_path, all_nb_primary_path):
    with open(all_karte_path, mode="rb") as f:
        all_karte = joblib.load(f)
    existed_primary_list = list(primary_images_dict.keys())
    extracted_karte = all_karte[all_karte['primary'].isin(existed_primary_list)]
    extracted_karte_count = extracted_karte['primary'].value_counts()
    non_abnormality_karte = extracted_karte[extracted_karte['data_label'] == '異常所見なし']
    non_abnormality_karte_count = non_abnormality_karte['primary'].value_counts()
    all_non_abnormality_primary_list = list()
    for primary, count in non_abnormality_karte_count.items():
        if extracted_karte_count[primary] == count:
            all_non_abnormality_primary_list.append(primary)
    print("all_primary_num:", len(extracted_karte_count))
    print("all_non_abnormality_num:", len(all_non_abnormality_primary_list))
    all_non_abnormality_primary_index = \
        extracted_karte.index[extracted_karte['primary'].isin(all_non_abnormality_primary_list)]
    extracted_karte = extracted_karte.drop(all_non_abnormality_primary_index)
    with open(extracted_karte_path, mode="wb") as f:
        joblib.dump(extracted_karte, f, compress=3)
    with open(all_nb_primary_path, mode="wb") as f:
        joblib.dump(all_non_abnormality_primary_list, f, compress=3)


def save_primary_images_dict(images_path, primary_images_path):
    images = glob.iglob(images_path, recursive=True)
    primary_images_dict = dict()

    for image in tqdm(images):
        if os.stat(image).st_size == 0:
            continue
        primary = int(image.split("/")[-2])
        if primary in primary_images_dict.keys():
            primary_images_dict[primary].append(image)
        else:
            primary_images_dict[primary] = [image]
    primary_images_dict = {i: sorted(list(set(j))) for i, j in primary_images_dict.items()}

    with open(primary_images_path, mode="wb") as f:
        joblib.dump(primary_images_dict, f, compress=3)
    return primary_images_dict


def save_primary_labels_dict(extracted_karte_path, primary_labels_path):
    with open(extracted_karte_path, mode="rb") as f:
        all_karte = joblib.load(f)
    print(all_karte.info())

    all_karte = all_karte[['data_label', 'primary']]
    primary = None
    labels = None
    primary_labels_dict = dict()

    for karte in all_karte.itertuples():
        if primary is None or primary != karte.primary:
            if primary is not None:
                if primary in primary_labels_dict.keys():
                    if labels:
                        primary_labels_dict[primary].extend(labels)
                else:
                    if labels:
                        primary_labels_dict[primary] = labels
            primary = karte.primary
            labels = list()
        if type(karte.data_label) == str:
            labels.append(karte.data_label)
    else:
        if primary in primary_labels_dict.keys():
            if labels:
                primary_labels_dict[primary].extend(labels)
        else:
            if labels:
                primary_labels_dict[primary] = labels

    print("primary length:", len(primary_labels_dict.keys()))
    print("labels length:", len(list(itertools.chain.from_iterable(primary_labels_dict.values()))))
    with open(primary_labels_path, mode="wb") as f:
        joblib.dump(primary_labels_dict, f, compress=3)


def create_data(primary_labels_path, all_nb_primary_path, label_hierarchical_data_path, label_data_path,
                label_vector_path, label_onehot_vector_path, label_nested_onehot_vector_path, num_classes_path, only_top):
    with open(primary_labels_path, mode="rb") as f:
        primary_labels_dict = joblib.load(f)
    with open(all_nb_primary_path, mode="rb") as f:
        all_nb_primary_list = joblib.load(f)
    label_hierarchical_data_dict = create_label_lib(primary_labels_dict, label_hierarchical_data_path)
    label_hierarchical_data_dict, label_data_dict, label_vector_dict = \
        vectorize(primary_labels_dict, label_hierarchical_data_dict,
                  choose_cut_line(label_hierarchical_data_dict),
                  label_hierarchical_data_path, label_data_path, label_vector_path, num_classes_path, only_top)
    label_onehot_vector_dict = \
        onehot_vectorize(label_data_dict, label_vector_dict, all_nb_primary_list, label_onehot_vector_path)
    label_nested_onehot_vector_dict = \
        nested_onehot_vectorize(label_data_dict, label_vector_dict, all_nb_primary_list, label_nested_onehot_vector_path)
    return label_onehot_vector_dict, label_nested_onehot_vector_dict


def create_dataset(primary_label_dict, images_path, image_label_path):
    images = glob.iglob(images_path, recursive=True)
    image_path_list = list()
    label_list = list()
    for image in tqdm(images):
        if os.stat(image).st_size == 0:
            print(image)
            continue
        target_patient = int(image.split("/")[-2])
        if target_patient not in primary_label_dict.keys():
            continue
        target_data = primary_label_dict[target_patient]
        image_path_list.append(image)
        label_list.append(target_data)

    image_label_dict = {
        'image_path': image_path_list,
        'label': label_list}
    with open(image_label_path, mode="wb") as f:
        joblib.dump(image_label_dict, f, compress=3)


def create_sequential_dataset(primary_label_dict, primary_images_dict, image_label_path):
    image_paths_list = list()
    label_list = list()
    for target_patient, image_paths in primary_images_dict.items():
        if target_patient not in primary_label_dict.keys():
            continue
        target_data = primary_label_dict[target_patient]
        image_paths_list.append(image_paths)
        label_list.append(target_data)

    image_label_dict = {
        'image_paths': image_paths_list,
        'label': label_list}
    with open(image_label_path, mode="wb") as f:
        joblib.dump(image_label_dict, f, compress=3)


def main(dataset_name, only_top=False, sequential=False, nested=False):
    file_path = Path(__file__)

    data_path = (file_path / '..' / '..' / 'data').resolve()
    kartes_path = str(data_path / '*' / '*.xlsx')

    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / dataset_name
    os.makedirs(dataset_path, exist_ok=True)

    all_karte_path = str(dataset_path / 'all_karte.joblib')
    save_concat_karte(kartes_path, all_karte_path)

    images_path = str(data_path / '**' / '[0-1][0-9]' / '[0-3][0-9]' / '*' / '*')
    primary_images_path = str(dataset_path / 'primary_images.joblib')
    primary_images_dict = save_primary_images_dict(images_path, primary_images_path)

    extracted_karte_path = str(dataset_path / 'extracted_karte.joblib')
    all_nb_primary_path = str(dataset_path / 'all_nb_primary.joblib')
    save_extracted_karte(all_karte_path, primary_images_dict, extracted_karte_path, all_nb_primary_path)

    primary_labels_path = str(dataset_path / 'primary_labels.joblib')
    save_primary_labels_dict(extracted_karte_path, primary_labels_path)

    label_hierarchical_data_path = str(dataset_path / 'label_hierarchical_data.joblib')
    label_data_path = str(dataset_path / 'label_data.joblib')
    label_vector_path = str(dataset_path / 'label_vector.joblib')
    label_onehot_vector_path = str(dataset_path / 'label_onehot_vector.joblib')
    label_nested_onehot_vector_path = str(dataset_path / 'label_onehot_vector.joblib')
    num_classes_path = str(dataset_path / 'num_classes.joblib')
    label_onehot_vector_dict, label_nested_onehot_vector_dict =\
        create_data(primary_labels_path, all_nb_primary_path, label_hierarchical_data_path, label_data_path,
                    label_vector_path, label_onehot_vector_path, label_nested_onehot_vector_path, num_classes_path, only_top)

    image_label_path = str(dataset_path / 'image_label.joblib')
    if sequential:
        if nested:
            create_sequential_dataset(label_nested_onehot_vector_dict, primary_images_dict, image_label_path)
        else:
            create_sequential_dataset(label_onehot_vector_dict, primary_images_dict, image_label_path)
    else:
        if nested:
            create_dataset(label_nested_onehot_vector_dict, images_path, image_label_path)
        else:
            create_dataset(label_onehot_vector_dict, images_path, image_label_path)

    with open(label_data_path, mode="rb") as f:
        label_data_dict = joblib.load(f)

    dataset_description_path = str(dataset_path / 'dataset_description.txt')
    with open(dataset_description_path, mode="w") as f:
        pprint(dataset_name)
        pprint('only_top: ' + str(only_top) + ', sequential: ' + str(sequential) + ', nested: ' + str(nested))

        pprint('top_cut_line: ' + str(label_data_dict['top_cut_line']), stream=f)
        pprint('sub_cut_line: ' + str(label_data_dict['sub_cut_line']), stream=f)
        pprint('len_top_label: ' + str(label_data_dict['len_top_label']), stream=f)
        pprint('len_sub_label: ' + str(label_data_dict['len_sub_label']), stream=f)
        pprint('len_all_label: ' + str(label_data_dict['len_all_label']), stream=f)
        pprint('top_label_list:', stream=f)
        pprint(label_data_dict['top_label_list'], stream=f)
        pprint('sub_label_list:', stream=f)
        pprint(label_data_dict['sub_label_list'], stream=f)
        pprint('top_sub_dict:', stream=f)
        pprint(label_data_dict['top_sub_dict'], stream=f)
        pprint('all_label_list:', stream=f)
        pprint(label_data_dict['all_label_list'], stream=f)


if __name__ == "__main__":
    main('dataset0')
    main('dataset1', only_top=True)
    main('dataset2', sequential=True, only_top=True)
