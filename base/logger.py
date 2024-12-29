from base.utils import ensure_dir, save_to_pickle, sigmoid
import os
import sys
import statistics
import numpy as np

from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score


# ------------------------------------------------------------------------------
# This file now contains only placeholders or helper functions you might use
# for binary classification logging. The classes and methods that dealt
# with continuous/regression labels have been removed.
# ------------------------------------------------------------------------------


class ClassificationLogger(object):
    """
    Example placeholder for a class to log binary classification metrics
    like accuracy, F1, confusion matrices, etc.

    You can expand or modify this class to suit your logging needs.
    """
    def __init__(self):
        # Example: store predictions and labels for an entire partition
        self.predictions = []
        self.labels = []

    def update(self, preds, labs):
        """
        Update logger with new predictions and labels.
        `preds` and `labs` should be lists or arrays of size (batch_size,).
        """
        self.predictions.extend(preds)
        self.labels.extend(labs)

    def compute_metrics(self):
        """
        Compute final metrics like accuracy or F1.
        """
        if not self.predictions or not self.labels:
            return {}

        acc = accuracy_score(self.labels, self.predictions)
        f1 = f1_score(self.labels, self.predictions, average='binary')
        return {
            'accuracy': acc,
            'f1_score': f1
        }

    def save_results(self, path):
        """
        Save predictions and labels to a pickle file or any format you prefer.
        """
        ensure_dir(path)
        data_to_save = {
            'predictions': np.array(self.predictions),
            'labels': np.array(self.labels)
        }
        save_to_pickle(path, data_to_save, replace=True)
