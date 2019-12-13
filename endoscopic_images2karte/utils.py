import numpy as np


def calc_f1score(running_mcm, multi):
    if multi:
        micro_mcm = running_mcm.sum(axis=0)
        macro_mcm = running_mcm
        with np.errstate(divide='ignore', invalid='ignore'):
            micro_precision = np.nan_to_num(micro_mcm[1][1] / micro_mcm.sum(axis=0)[1])
            micro_recall = np.nan_to_num(micro_mcm[1][1] / micro_mcm.sum(axis=1)[1])
            micro_f1score = np.nan_to_num(2 * (micro_precision * micro_recall) / (micro_precision + micro_recall))
            macro_precision = np.nan_to_num(macro_mcm[:, 1, 1] / macro_mcm.sum(axis=1)[:, 1])
            macro_recall = np.nan_to_num(macro_mcm[:, 1, 1] / macro_mcm.sum(axis=2)[:, 1])
            macro_f1score = np.nan_to_num(2 * (macro_precision * macro_recall) / (macro_precision + macro_recall))
        f1score_dict = {
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1score': micro_f1score,
            'macro_precision': macro_precision.mean(),
            'macro_recall': macro_recall.mean(),
            'macro_f1score': macro_f1score.mean()
        }
    else:
        mcm = running_mcm
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.nan_to_num(mcm[1][1] / mcm.sum(axis=0)[1])
            recall = np.nan_to_num(mcm[1][1] / mcm.sum(axis=1)[1])
            f1score = np.nan_to_num(2 * (precision * recall) / (precision + recall))
        f1score_dict = {
            'precision': precision,
            'recall': recall,
            'f1score': f1score,
        }

    return f1score_dict
