from collections import Counter
import itertools
from typing import Dict

import joblib
from pathlib import Path
import torch
from tqdm import tqdm

from text_process import text_process


def create_label_lib(primary_labels_dict, label_hierarchical_data_path):
    label_list = list(set(itertools.chain.from_iterable(primary_labels_dict.values())))
    print("pre process label list length:", len(label_list))
    new_label_list = list()
    for label in label_list:
        for new_label in text_process(label, specific=True):
            new_label = sorted(set(new_label), key=new_label.index)
            new_label_list.append(new_label)
    new_label_list = sorted(list(map(list, set(map(tuple, new_label_list)))), key=lambda x: len(x), reverse=True)
    print("processed label list length:", len(new_label_list))
    top_label = list()
    top_count_label = list()
    sub_label = list()
    sub_count_label = list()
    top_sub_dict: Dict[int, list] = dict()
    for label in new_label_list:
        top_index = None
        sub_index_list = list()
        for id, word in enumerate(label):
            if id == 0:
                top_count_label.append(word)
                if word in top_label:
                    top_index = top_label.index(word)
                else:
                    top_label.append(word)
                    top_index = len(top_label)
            else:
                sub_count_label.append(word)
                if word in sub_label:
                    sub_index_list.append(sub_label.index(word))
                else:
                    sub_label.append(word)
                    sub_index_list.append(len(sub_label))
        if top_index in top_sub_dict.keys():
            for sub_index in sub_index_list:
                if sub_index not in top_sub_dict[top_index]:
                    top_sub_dict[top_index].append(sub_index)
        else:
            top_sub_dict[top_index] = sub_index_list
    top_count_label = Counter(top_count_label)
    sub_count_label = Counter(sub_count_label)
    print("top label list length:", len(top_label))
    print("sub label list length:", len(sub_label))
    top_count = dict()
    for count in top_count_label.values():
        if count in top_count.keys():
            top_count[count] += 1
        else:
            top_count[count] = 1

    print(sorted(top_count.items(), key=lambda x: x[0]))

    sub_count = dict()
    for count in sub_count_label.values():
        if count in sub_count.keys():
            sub_count[count] += 1
        else:
            sub_count[count] = 1
    print(sorted(sub_count.items(), key=lambda x: x[0]))

    label_hierarchical_data_dict = {
        "top_label": top_label,
        "sub_label": sub_label,
        "top_sub_dict": top_sub_dict,
        "top_count_label": top_count_label,
        "sub_count_label": sub_count_label,
        "top_count": top_count,
        "sub_count": sub_count
    }
    with open(label_hierarchical_data_path, mode="wb") as f:
        joblib.dump(label_hierarchical_data_dict, f, compress=3)
    return label_hierarchical_data_dict


def choose_cut_line(label_hierarchical_data_dict=None, label_hierarchical_data_path=None, default_cut_line=5):
    if label_hierarchical_data_dict is None:
        with open(label_hierarchical_data_path, mode="rb") as f:
            label_hierarchical_data_dict = joblib.load(f)
    num_top_label = len(label_hierarchical_data_dict["top_label"])
    num_sub_label = len(label_hierarchical_data_dict["sub_label"])
    top_count = label_hierarchical_data_dict["top_count"]
    sub_count = label_hierarchical_data_dict["sub_count"]

    top_cut_line_dict = dict()
    print("top label num:", num_top_label)
    for count, num_label in sorted(top_count.items(), key=lambda x: x[0]):
        num_top_label -= num_label
        print("cut line " + str(count) + ": ", num_top_label)
        top_cut_line_dict[count] = num_top_label
    try:
        top_cut_line = int(input("choose top label cut line: "))
    except ValueError:
        top_cut_line = default_cut_line
    top_label_num = top_cut_line_dict[top_cut_line]

    sub_cut_line_dict = dict()
    print("sub label num:", num_sub_label)
    for count, num_label in sorted(sub_count.items(), key=lambda x: x[0]):
        num_sub_label -= num_label
        print("cut line " + str(count) + ": ", num_sub_label)
        sub_cut_line_dict[count] = num_sub_label
    try:
        sub_cut_line = int(input("choose sub label cut line: "))
    except ValueError:
        sub_cut_line = default_cut_line
    sub_label_num = sub_cut_line_dict[sub_cut_line]

    return top_cut_line, sub_cut_line, top_label_num, sub_label_num


def vectorize(primary_labels_dict, label_hierarchical_data_dict, cut_line_info,
              label_hierarchical_data_path, label_data_path, label_vector_path, num_classes_path, only_top):
    label_vector_dict = dict()
    top_label = label_hierarchical_data_dict["top_label"]
    sub_label = label_hierarchical_data_dict["sub_label"]
    top_count_label = label_hierarchical_data_dict["top_count_label"]
    sub_count_label = label_hierarchical_data_dict["sub_count_label"]
    top_other_list = list()
    sub_other_list = list()

    top_cut_line, sub_cut_line, top_label_num, sub_label_num = cut_line_info

    len_top_label = top_label_num
    len_sub_label = sub_label_num
    top_index_head = 1
    sub_index_head = len_top_label + 1

    if only_top:
        len_all_label = len_top_label + 1
    else:
        len_all_label = len_top_label + len_sub_label + 1
    with open(num_classes_path, mode="wb") as f:
        joblib.dump(len_all_label, f, compress=3)

    label_data_dict = dict()
    label_data_dict['top_cut_line'] = top_cut_line
    label_data_dict['sub_cut_line'] = sub_cut_line
    label_data_dict['len_top_label'] = len_top_label
    label_data_dict['len_sub_label'] = len_sub_label
    label_data_dict['len_all_label'] = len_all_label

    top_label_list = list()
    sub_label_list = list()
    top_sub_dict: Dict[int, list] = dict()

    for primary, label_list in tqdm(primary_labels_dict.items()):
        label_vector = [0]
        for label in label_list:
            for new_label in text_process(label, specific=True):
                new_label = sorted(set(new_label), key=new_label.index)
                top_index = None
                sub_index_list = list()
                for id, word in enumerate(new_label):
                    if id == 0:
                        if top_count_label[word] > top_cut_line:
                            if word in top_label_list:
                                top_index = top_label_list.index(word)
                            else:
                                top_index = len(top_label_list)
                                top_label_list.append(word)
                            top_label_index = top_index_head + top_index
                            if top_label_index not in label_vector:
                                label_vector.append(top_label_index)
                        else:
                            top_other_index = top_label.index(word)
                            if top_other_index not in top_other_list:
                                top_other_list.append(top_other_index)
                    elif not only_top:
                        if sub_count_label[word] > sub_cut_line:
                            if word in sub_label_list:
                                sub_index = sub_label_list.index(word)
                            else:
                                sub_index = len(sub_label_list)
                                sub_label_list.append(word)
                            sub_index_list.append(sub_index)
                            sub_label_index = sub_index_head + sub_index
                            if sub_label_index not in label_vector:
                                label_vector.append(sub_label_index)
                        else:
                            sub_other_index = sub_label.index(word)
                            if sub_other_index not in sub_other_list:
                                sub_other_list.append(sub_other_index)
                if not only_top and top_index is not None and len(sub_index_list) != 0:
                    if top_index in top_sub_dict.keys():
                        for sub_index in sub_index_list:
                            if sub_index not in top_sub_dict[top_index]:
                                top_sub_dict[top_index].append(sub_index)
                    else:
                        top_sub_dict[top_index] = sub_index_list
        label_vector_dict[primary] = sorted(list(set(label_vector)))

    assert len(top_label_list) == len_top_label
    if not only_top:
        assert len(sub_label_list) == len_sub_label
    all_label_list = top_label_list + sub_label_list
    label_data_dict['top_label_list'] = top_label_list
    label_data_dict['sub_label_list'] = sub_label_list
    label_data_dict['top_sub_dict'] = top_sub_dict
    if only_top:
        label_data_dict['all_label_list'] = top_label_list
    else:
        label_data_dict['all_label_list'] = all_label_list

    label_hierarchical_data_dict["top_other_list"] = sorted(list(set(top_other_list)))
    label_hierarchical_data_dict["sub_other_list"] = sorted(list(set(sub_other_list)))

    with open(label_hierarchical_data_path, mode="wb") as f:
        joblib.dump(label_hierarchical_data_dict, f, compress=3)
    with open(label_data_path, mode="wb") as f:
        joblib.dump(label_data_dict, f, compress=3)
    with open(label_vector_path, mode="wb") as f:
        joblib.dump(label_vector_dict, f, compress=3)
    return label_hierarchical_data_dict, label_data_dict, label_vector_dict


def onehot_vectorize(label_data_dict, label_vector_dict, all_nb_primary_list, label_onehot_vector_path):
    label_onehot_vector_dict = dict()
    len_all_label = label_data_dict['len_all_label']
    for primary, labels in label_vector_dict.items():
        labels = torch.zeros(len_all_label).scatter(0, torch.tensor(labels), 1.)
        label_onehot_vector_dict[primary] = labels
    for primary in all_nb_primary_list:
        labels = torch.zeros(len_all_label)
        label_onehot_vector_dict[primary] = labels
    with open(label_onehot_vector_path, mode="wb") as f:
        joblib.dump(label_onehot_vector_dict, f, compress=3)
    return label_onehot_vector_dict


def nested_onehot_vectorize(label_data_dict, label_vector_dict, all_nb_primary_list, label_nested_onehot_vector_path):
    label_nested_onehot_vector_dict = dict()
    len_all_label = label_data_dict['len_all_label']
    for primary, labels in label_vector_dict.items():
        labels = torch.tensor(labels)
        labels = labels.unsqueeze(0)
        labels = torch.zeros(labels.size(0), len_all_label).scatter(1, labels, 1.)
        label_nested_onehot_vector_dict[primary] = labels
    for primary in all_nb_primary_list:
        labels = torch.zeros(1)
        labels = labels.unsqueeze(0)
        labels = torch.zeros(labels.size(0), len_all_label)
        label_nested_onehot_vector_dict[primary] = labels
    with open(label_nested_onehot_vector_path, mode="wb") as f:
        joblib.dump(label_nested_onehot_vector_dict, f, compress=3)
    return label_nested_onehot_vector_dict


if __name__ == "__main__":
    file_path = Path(__file__)
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    dataset_path = datasets_path / 'dataset0'
    primary_labels_path = str(dataset_path / 'primary_labels.joblib')
    with open(primary_labels_path, mode="rb") as f:
        primary_labels_dict = joblib.load(f)
    label_hierarchical_data_path = str(dataset_path / 'label_hierarchical_data.joblib')
    label_hierarchical_data_dict = create_label_lib(primary_labels_dict, label_hierarchical_data_path)

    label_vector_path = str(dataset_path / 'label_vector.joblib')
    label_data_path = str(dataset_path / 'label_data.joblib')
    label_hierarchical_data_dict, label_data_dict, label_vector_dict = \
        vectorize(primary_labels_dict, label_hierarchical_data_dict, choose_cut_line(label_hierarchical_data_dict),
              label_hierarchical_data_path, label_data_path, label_vector_path)
    for label in label_vector_dict.values():
        print(label)
        print(len(label))
        break
    for data in label_data_dict.values():
        print(data)

    label_onehot_vector_path = str(dataset_path / 'label_onehot_vector.joblib')
    label_onehot_vector_dict = \
        onehot_vectorize(label_data_dict, label_vector_dict, label_onehot_vector_path)
    # print(label_onehot_vector_dict)
    for label in label_onehot_vector_dict.values():
        print(label)
        print(label.size())
        break
