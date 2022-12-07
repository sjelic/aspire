#dataset_meta_json = torch.load(dataset_meta_json_path)
import torch
import logging
from random import sample
from RapidECalibrationDataset import RapidECalibrationDataset
from Splitter import Splitter
from Standardize import RapidECalibrationStandardize

class StratifiedSplitterForRapidECalibDataset(Splitter):
    def __init__(self, N, dataset = None):
        super(StratifiedSplitterForRapidECalibDataset, self).__init__()
        self.N = N
        if (isinstance(dataset, RapidECalibrationDataset)):
            self.dataset = dataset
        else:
            logging.error('Dataset should be an instance of StratifiedSplitterForRapidECalibDat. EXIT.')
            exit(0)

    def __call__(self):
        splits = sample(list(range(len(self.dataset.train_splits))), self.N)
        splits.sort()
        splits = [(self.dataset.train_splits[i],torch.load(self.dataset.train_splits[i])['pair_split_path']) for i in range(self.N)]
        return [(RapidECalibrationDataset(meta_json=path[0],transform=RapidECalibrationStandardize(path[0])), RapidECalibrationDataset(meta_json=path[1],transform=RapidECalibrationStandardize(path[0]))) for path in splits]

