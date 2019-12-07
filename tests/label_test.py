from pathlib import Path
import joblib


def test(flag_label_list, labels_file, label_list_file=None):
    file_path = Path(__file__)
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    labels_path = (datasets_path / 'labels').resolve()
    labels_file = str(labels_path / labels_file)
    with open(labels_file, mode="rb") as f:
        primary_label_dict = joblib.load(f)
    count = 0
    for primary, label in primary_label_dict.items():
        print(primary)
        print(label)
        print()
        count += 1
        if count > 10:
            break

    if flag_label_list:
        label_list_file = str(labels_path / label_list_file)
        with open(label_list_file, mode="rb") as f:
            label_list = joblib.load(f)
        count = 0
        for label in label_list:
            print(label)
            count += 1
            if count > 10:
                break



if __name__ == "__main__":
    print("test0")
    test(flag_label_list=False, labels_file='labels0.joblib')
    print("test1")
    test(flag_label_list=True, labels_file='labels1.joblib', label_list_file='labels1list.joblib')
    print("test2")
    test(flag_label_list=True, labels_file='labels2.joblib', label_list_file='labels2list.joblib')
    print("test3")
    test(flag_label_list=True, labels_file='labels3.joblib', label_list_file='labels3list.joblib')
