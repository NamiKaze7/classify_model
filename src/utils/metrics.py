from typing import Set, Tuple, Union, List

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


class MetricsF1(object):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computes exact match and F1 score using the official DROP
    evaluator (which has special handling for numbers and for questions with multiple answer spans,
    among other things).
    """

    def __init__(self, num_tags=2, mode=None) -> None:
        self.num_tags = num_tags
        self._total_TP = np.array([0] * num_tags)
        self._total_FN = np.array([1] * num_tags)
        self._total_FP = np.array([1] * num_tags)
        self._count = 0
        self._total_acc = 0
        self._details = []

    def __call__(self, gold, prediction, raw_text, prob):
        for i in range(len(prediction)):
            label = int(gold[i])
            pred = int(prediction[i])
            score = float(prob[i][label])
            sp = raw_text[i]
            if label == pred:
                self._total_TP[label] += 1
            else:
                self._total_FN[label] += 1
                self._total_FP[pred] += 1
            self._count += 1
            self._total_acc += pred == label
            it = {'raw_text': sp,
                  'acc': pred == label,
                  'label': label,
                  'pred': pred,
                  'score': score
                  }
            self._details.append(it)

    def get_overall_metric(self):
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official DROP script
        over all inputs.
        """
        precision = (self._total_TP / (self._total_TP + self._total_FP)).mean()
        recall = (self._total_TP / (self._total_TP + self._total_FN)).mean()
        f1 = 2 * precision * recall / (precision + recall)
        acc = self._total_acc / self._count
        df = pd.DataFrame([precision, recall, f1, acc], index=['precision', 'recall', 'f1', 'acc'], columns=['score'])
        metrics = {'p': precision, 'r': recall, 'f1': f1, 'acc': acc, 'dataframe': df}
        return metrics

    def get_raw(self):
        return pd.DataFrame(self._details)

    def reset(self):
        self._total_TP = np.array([0] * self.num_tags)
        self._total_FN = np.array([1] * self.num_tags)
        self._total_FP = np.array([1] * self.num_tags)
        self._count = 0
        self._total_acc = 0
        self._details = []

    def __str__(self):
        return f"MetricsF1(count={self._count})"
