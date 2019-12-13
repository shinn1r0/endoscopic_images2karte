from pathlib import Path
import joblib
import re
import pandas as pd
from janome.analyzer import Analyzer
from janome.tokenfilter import POSKeepFilter


ZEN = "".join(chr(0xff01 + i) for i in range(94))
HAN = "".join(chr(0x21 + i) for i in range(94))
ZEN2HAN = str.maketrans(ZEN, HAN)
HAN2ZEN = str.maketrans(HAN, ZEN)
token_filters = [POSKeepFilter(['助詞', '接続詞', '接頭辞', '助動詞', '名詞,副詞可能'])]
a = Analyzer(token_filters=token_filters)
regex_delimiter = re.compile(",+|(^[0-9])+\.+|、+|。+|・+|･+|;+|:+| +|　+|\n+|\t+|\r+|に対して|による|にて")
regex_or = re.compile("[ァ-ヴ]+or[ァ-ヴ]+")
regex_symbol = re.compile('\++|-+|±+|\+\++')
regex_cut = [
    re.compile('[0-9]*\.*[0-9]*[a-z]*m[a-z]*')
]
regex_remove = [
    re.compile('(所見なし|所見はっきりせず|所見なく)')
]


def text_process(label, specific=True):
    label = str(label)
    label = label.translate(ZEN2HAN)
    # label = re.split(",+|\.+|、+|。+|・+|･+| +|　+|([^a-z]+or[^a-z]+)+|\n+|\t+|\r+", label)
    label = regex_delimiter.split(label)
    new_label = [x for x in label if x]
    if len(new_label) == 1 and new_label[0] == "異常所見なし":
        return new_label
    if specific:
        new_label_list = text_specific_process_wrap(new_label)
    else:
        new_label_list = [new_label]
    for new_label in new_label_list:
        if new_label:
            yield new_label


def text_specific_process_wrap(label_list):
    new_label_list = list()
    if label_list[0] == '慢性胃炎':
        new_label = list()
        for label in label_list:
            if len(regex_symbol.findall(label)) > 1:
                start = 0
                for i, l in enumerate(label):
                    if regex_symbol.search(l) is not None and i != len(label):
                        new_label.append(label[start:i+1])
                        start = i+1
                else:
                    new_label.append(label[start:])
            else:
                new_label.append(label)
    else:
        new_label = label_list
    new_label = text_specific_process(new_label)
    if new_label.count('ヘルニア'):
        label_index = new_label.index('ヘルニア')
        if label_index > 0:
            new_label_list.append(new_label[:label_index])
            new_label_list.append(new_label[label_index:])
        else:
            new_label_list.append(new_label)
    elif new_label.count('ポリープ'):
        label_index = new_label.index('ポリープ')
        if label_index > 0:
            new_label_list.append(new_label[:label_index])
            new_label_list.append(new_label[label_index:])
        else:
            new_label_list.append(new_label)
    else:
        new_label_list.append(new_label)
    return new_label_list


def text_specific_process(label_list):
    new_label = list()
    target_stack = ''
    former_specific_disease_list = [
        '食道癌', 'マロリーワイス症候群', 'ヘルニア', 'ポリープ', 'カンジダ症', 'ヨード不染', '咽頭喉頭炎',
        '十二指腸炎', '静脈拡張', '潰瘍', 'リンパ腫', 'ペンタサ'
    ]
    chronic_gastritis_list = ['RAC', '胃底腺ポリープ', '稜線状発赤', '扁平隆起', 'ヘマチン付着', '地図状発赤', '色調逆転現症']
    symbols = ['±', '+', '-', '++']
    for label in label_list:
        label = label.replace('疑う所見なし', '所見なし')
        label = label.replace('示唆する所見なし', '所見なし')
        if any([(rl.search(label) is not None) for rl in regex_remove]):
            continue
        label = label.replace('CRTX', 'CRTx')
        label = label.replace('no', '')
        label = label.replace('interval', '')
        label = label.replace('change', '')
        label = label.replace('polyp', 'ポリープ')
        label = label.replace('あり', '')
        label = label.replace('有り', '')
        label = label.replace('有', '')
        label = label.replace('所見', '')
        label = label.replace('疑う', '疑い')
        if bool(target_stack):
            if 'chronic_gastritis' in target_stack:
                if any([bool(chronic_gastritis in target_stack) for chronic_gastritis in chronic_gastritis_list]):
                    label = label.replace('ー', '-')
                    for chronic_gastritis in chronic_gastritis_list:
                        if chronic_gastritis in target_stack:
                            for symbol in symbols:
                                if symbol in label:
                                    label_append(new_label, chronic_gastritis + symbol)
                                    label_append(new_label, target_stack.replace('chronic_gastritis', '')
                                                 .replace(chronic_gastritis, ''))
                                    target_stack = 'chronic_gastritis'
                    if target_stack == 'chronic_gastritis':
                        continue
                    else:
                        target_stack = 'chronic_gastritis'

                if '性胃炎' in label:
                    label_append(new_label, label)

                if any([bool(chronic_gastritis in label) for chronic_gastritis in chronic_gastritis_list]):
                    for chronic_gastritis in chronic_gastritis_list:
                        if chronic_gastritis in label:
                            target_stack += label
                elif 'びらん' in label:
                    if 'びらん性' not in label:
                        label_append(new_label, 'びらん')
                        if '散在' in label:
                            label_append(new_label, label.replace('散在', '').replace('びらん', ''))
                            label_append(new_label, '散在')
                        else:
                            label_append(new_label, label.replace('びらん', ''))
                else:
                    label_append(new_label, label)
            if 'esophageal_varices' in target_stack:
                pass
            if 'after_gastric_surgery' in target_stack:
                if 'cut_method' in target_stack:
                    pass
                else:
                    label_index = label.find('(')
                    if label_index > 0:
                        label_append(new_label, label[:label_index])
                        target_stack = 'after_gastric_surgery_cut_method'
                    if label.find(')') > 0:
                        target_stack = 'after_gastric_surgery'
                    else:
                        label_append(new_label, label)
            if 'reflux_esophagitis' in target_stack:
                if 'grade' in target_stack:
                    label_append(new_label, 'grade ' + label)
                    target_stack = 'reflux_esophagitis'
                elif label.find('grade') != -1:
                    target_stack = 'reflux_esophagitis_grade'
                elif label.find('LA') != -1:
                    label_append(new_label, 'LA')
                    target_stack = ''
            continue

        if not label or label == 'nan':
            pass
        elif 'CRT' in label:
            label_append(new_label, label.replace('CRTx後', ''))
            label_append(new_label, 'CRTx後')
        elif 'ELPS後' in label:
            label_append(new_label, label.replace('瘢痕', '').replace('ELPS後', ''))
            label_append(new_label, 'ELPS後瘢痕')
        elif '軽度' in label:
            label_append(new_label, label.replace('軽度', ''))
            label_append(new_label, '軽度')
        elif '慢性胃炎' in label:
            label_append(new_label, label)
            target_stack = 'chronic_gastritis'
        # elif '食道静脈瘤' in label:
        #     label_append(new_label, label)
        #     target_stack = 'esophageal_varices'
        elif '胃術後' in label:
            label_append(new_label, label)
            target_stack = 'after_gastric_surgery'
        elif '癌術後' in label:
            label_append(new_label, label.replace('術後', ''))
            label_append(new_label, '術後')
        elif '逆流性食道炎' in label:
            label_append(new_label, label)
            target_stack = 'reflux_esophagitis'
        elif any([(rc.search(label) is not None) for rc in regex_cut]):
            for rc in regex_cut:
                cut_list = rc.findall(label)
                for cl in cut_list:
                    label_append(new_label, cl)
                    label_append(new_label, label.replace(cl, ''))
        elif any([(fd in label) for fd in former_specific_disease_list]):
            for fd in former_specific_disease_list:
                if fd in label:
                    label_append(new_label, fd)
                    label_append(new_label, label.replace(fd, ''))
        elif 'びらん' in label:
            if 'びらん性' not in label:
                label_append(new_label, 'びらん')
                label_append(new_label, label.replace('びらん', ''))
        elif '疑い' in label:
            label_append(new_label, label.replace('疑い', ''))
            label_append(new_label, '疑い')
        else:
            if len(label) < 20:
                if label.find('(') != -1:
                    label_append(new_label, label)
                else:
                    label_append(new_label, label)

    new_label = [x for x in new_label if x]
    return new_label


def label_append(label_list, label):
    if regex_or.search(label) is not None:
        for ll in re.split('or', label):
            if not label_analyzer(ll):
                label_list.append(ll.replace('(', '').replace(')', ''))
    elif not label_analyzer(label):
        label_list.append(label.replace('(', '').replace(')', ''))


def label_analyzer(label):
    label = label.replace('術後', '')
    label = label.replace('の過形成', '')
    tokens = list(a.analyze(label))
    return bool(tokens)


def main(labels_file, label_list_file=None):
    file_path = Path(__file__)
    datasets_path = (file_path / '..' / '..' / 'datasets').resolve()
    labels_path = (datasets_path / 'labels').resolve()
    labels_file = str(labels_path / labels_file)
    # with open(labels_file, mode="rb") as f:
    #     primary_label_dict = joblib.load(f)
    # count = 0
    # for primary, label in primary_label_dict.items():
    #     print(primary)
    #     print(label)
    #     print()
    #     count += 1
    #     if count > 10:
    #         break

    label_list_file = str(labels_path / label_list_file)
    with open(label_list_file, mode="rb") as f:
        label_list = joblib.load(f)
    print(len(label_list))

    label_list_csv = str(labels_path / 'label_list.csv')
    label_series = pd.DataFrame(label_list)
    label_series.to_csv(label_list_csv, index=False, header=False)

    new_label_list = list()
    for index, label in enumerate(label_list):
        for new_label in text_process(label, specific=False):
            new_label = [len(new_label)] + new_label
            new_label_list.append(sorted(set(new_label), key=new_label.index))
    new_label_list.sort(key=lambda x: x[0], reverse=True)

    new_label_list1_csv = str(labels_path / 'new_label_list1.csv')
    new_label_series1 = pd.DataFrame(new_label_list)
    new_label_series1.to_csv(new_label_list1_csv, index=False, header=False)

    new_label_list = list()
    for index, label in enumerate(label_list):
        for new_label in text_process(label, specific=True):
            new_label = [len(new_label)] + new_label
            new_label_list.append(sorted(set(new_label), key=new_label.index))
    new_label_list.sort(key=lambda x: x[0], reverse=True)

    new_label_list2_csv = str(labels_path / 'new_label_list2.csv')
    new_label_series = pd.DataFrame(new_label_list)
    new_label_series.to_csv(new_label_list2_csv, index=False, header=False)

    new_label_list = list()
    for index, label in enumerate(label_list):
        for new_label in text_process(label, specific=True):
            if len(new_label) > 0:
                new_label = [len(new_label)] + new_label
                new_label_list.append(sorted(set(new_label), key=new_label.index))
    new_label_list = sorted(list(map(list, set(map(tuple, new_label_list)))), key=lambda x: x[0], reverse=True)

    new_label_list3_csv = str(labels_path / 'new_label_list3.csv')
    new_label_series = pd.DataFrame(new_label_list)
    new_label_series.to_csv(new_label_list3_csv, index=False, header=False)


if __name__ == "__main__":
    main(labels_file='labels1.joblib', label_list_file='labels1list.joblib')
