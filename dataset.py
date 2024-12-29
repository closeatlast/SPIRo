from base.dataset import GenericDataArranger, GenericDataset
from torch.utils.data import Dataset as PytorchDataset

import numpy as np
from PIL import Image


class Dataset(GenericDataset):
    """
    Dataset class for binary classification tasks (PTSD vs. non-PTSD).
    Inherits from GenericDataset but removes 'continuous_label_dim'.
    """
    def __init__(self,
                 data_list,
                 modality,
                 multiplier,
                 feature_dimension,
                 window_length,
                 mode,
                 mean_std=None,
                 time_delay=0,
                 feature_extraction=0):
        # Call parent constructor WITHOUT 'continuous_label_dim'
        super().__init__(
            data_list=data_list,
            modality=modality,
            multiplier=multiplier,
            feature_dimension=feature_dimension,
            window_length=window_length,
            mode=mode,
            mean_std=mean_std,
            time_delay=time_delay,
            feature_extraction=feature_extraction
        )


class DataArranger(GenericDataArranger):
    """
    DataArranger for partitioning the dataset into train/validate/test sets.
    Also defines which features to load (audio, visual).
    """
    def __init__(self, dataset_info, dataset_path, debug):
        super().__init__(dataset_info, dataset_path, debug)

    @staticmethod
    def get_feature_list():
        """
        Return only the features you actually want.
        For pure audio+visual, you might keep 'vggish' (audio) and 'egemaps', etc.
        If you no longer use text embeddings, remove 'bert'.
        """
        feature_list = ['vggish', 'egemaps']
        return feature_list

    def partition_range_fn(self):
        """
        Defines which indices belong to train/validate/test/extra partitions.
        Adjust as needed. 
        """
        partition_range = {
            'train': [
                np.arange(0, 71),
                np.arange(71, 142),
                np.arange(142, 213),
                np.arange(213, 284),
                np.arange(284, 356)
            ],
            'validate': [np.arange(356, 432)],
            'test': [],
            'extra': [np.arange(432, 594)]
        }

        # If debug == 1, use a tiny partition for quick tests
        if self.debug == 1:
            partition_range = {
                'train': [np.arange(0, 1), np.arange(1, 2), np.arange(2, 3),
                          np.arange(3, 4), np.arange(4, 5)],
                'validate': [np.arange(5, 6)],
                'test': [np.arange(6, 7)],
                'extra': [np.arange(7, 8)]
            }
        return partition_range

    @staticmethod
    def assign_fold_to_partition():
        """
        Specifies how many folds correspond to each partition.
        E.g., 'train': 5 means 5 folds go into training, etc.
        """
        fold_to_partition = {
            'train': 5,
            'validate': 1,
            'test': 0,
            'extra': 1
        }
        return fold_to_partition
