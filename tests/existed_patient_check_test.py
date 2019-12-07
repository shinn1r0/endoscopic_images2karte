import glob
import os
from pathlib import Path

import joblib
import pandas as pd
from tqdm import tqdm


def save_concat_karte(kartes_path, all_karte_path):
    kartes = glob.glob(kartes_path)
    karte_list = list()
    for karte in kartes:
        karte_data = pd.read_excel(karte)
        if len(karte_data.columns) >= 10:
            karte_data = karte_data.rename(columns={
                '検査プライマリ': 'primary',
                '患者ID': 'patient_id',
                '質的診断': 'data_label'})
            karte_list.append(karte_data)
    all_karte = pd.concat(karte_list, sort=False)
    print(all_karte['patient_id'])
    all_karte = all_karte.dropna(subset=['primary']).astype({'primary': int})
    all_karte = all_karte.drop_duplicates(keep='first')
    print(all_karte.info())

    with open(all_karte_path, mode="wb") as f:
        joblib.dump(all_karte, f, compress=3)


def save_primary_patient_id_dict(all_karte_path, primary_patient_id_path):
    with open(all_karte_path, mode="rb") as f:
        all_karte = joblib.load(f)

    all_karte = all_karte[['data_label', 'primary', 'patient_id']]
    primary = None
    primary_patient_id_dict = dict()

    for karte in all_karte.itertuples():
        if primary is None or primary != karte.primary:
            if primary is not None:
                if primary in primary_patient_id_dict.keys():
                    assert primary_patient_id_dict[primary] == karte.patient_id
                else:
                    if karte.patient_id is not None:
                        primary_patient_id_dict[primary] = str(karte.patient_id)
            primary = karte.primary
    else:
        if primary is not None:
            if primary in primary_patient_id_dict.keys():
                assert primary_patient_id_dict[primary] == str(karte.patient_id)
            else:
                if karte.patient_id is not None:
                    primary_patient_id_dict[primary] = str(karte.patient_id)

    with open(primary_patient_id_path, mode="wb") as f:
        joblib.dump(primary_patient_id_dict, f, compress=3)

    return primary_patient_id_dict


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


def save_exited_patient(primary_patient_id_dict, primary_images_dict, existed_patient_path):
    existed_patient_list = list()
    for target_patient in primary_images_dict.keys():
        if target_patient in primary_patient_id_dict.keys():
            existed_patient_list.append(primary_patient_id_dict[target_patient])
    with open(existed_patient_path, mode="wb") as f:
        joblib.dump(existed_patient_list, f, compress=3)
    print(existed_patient_list)
    print(len(existed_patient_list))


def get_existed_patient(dataset_name):
    file_path = Path(__file__)

    data_path = (file_path / '..' / '..' / 'data').resolve()
    kartes_path = str(data_path / '*' / '*.xlsx')

    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / dataset_name
    os.makedirs(dataset_path, exist_ok=True)

    all_karte_path = str(dataset_path / 'all_karte.joblib')
    save_concat_karte(kartes_path, all_karte_path)
    primary_patient_id_path = str(dataset_path / 'primary_patient_id.joblib')
    primary_patient_id_dict = save_primary_patient_id_dict(all_karte_path, primary_patient_id_path)

    images_path = str(data_path / '**' / '[0-1][0-9]' / '[0-3][0-9]' / '*' / '*')
    primary_images_path = str(dataset_path / 'primary_images.joblib')
    primary_images_dict = save_primary_images_dict(images_path, primary_images_path)

    exited_patient_path = str(dataset_path / 'existed_patient.joblib')
    save_exited_patient(primary_patient_id_dict, primary_images_dict, exited_patient_path)


def save_removed_patient_csv(dataset_name, data_name):
    file_path = Path(__file__)
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / dataset_name
    exited_patient_path = str(dataset_path / 'existed_patient.joblib')
    with open(exited_patient_path, mode="rb") as f:
        exited_patient_list = joblib.load(f)

    data_path = (file_path / '..' / '..' / 'data').resolve()
    csv_path = str(data_path / data_name / '*.csv')
    csv_files = glob.glob(csv_path)
    for csv_file in csv_files:
        original_csv_file = csv_file[:-4] + '_original' + csv_file[-4:]
        os.rename(csv_file, original_csv_file)
        csv_data = pd.read_csv(original_csv_file, encoding='shift_jis')
        print(csv_data)

        drop_index_list = list()
        for csv_row in csv_data.itertuples():
            if str(csv_row[1]) in exited_patient_list:
                drop_index_list.append(csv_row[0])

        new_csv_data = csv_data.drop(csv_data.index[drop_index_list])
        print(new_csv_data)
        new_csv_data.to_csv(csv_file, index=False, encoding='shift_jis')


if __name__ == "__main__":
    # get_existed_patient('patient_check')
    save_removed_patient_csv('patient_check', 'existed_patient_check')
