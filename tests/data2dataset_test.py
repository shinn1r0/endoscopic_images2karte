import test_path_setting
from data2dataset import *


def test_save_concat_karte():
    file_path = Path(__file__)

    data_path = (file_path / '..' / '..' / 'data').resolve()
    kartes_path = str(data_path / '*' / '*.xlsx')
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / 'tests'
    os.makedirs(dataset_path, exist_ok=True)

    all_karte_path = str(dataset_path / 'all_karte.joblib')
    save_concat_karte(kartes_path, all_karte_path)

    images_path = str(data_path / '**' / '[0-1][0-9]' / '[0-3][0-9]' / '*' / '*')
    primary_images_path = str(dataset_path / 'primary_images.joblib')
    primary_images_dict = save_primary_images_dict(images_path, primary_images_path)

    extracted_karte_path = str(dataset_path / 'extracted_karte.joblib')
    all_nb_primary_path = str(dataset_path / 'all_nb_primary.joblib')
    save_extracted_karte(all_karte_path, primary_images_dict, extracted_karte_path, all_nb_primary_path)
