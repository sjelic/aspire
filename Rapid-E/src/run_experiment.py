import torch
import numpy as np
from torch import nn
from scipy import stats
import json
import logging
import sys, getopt
from Sampler import StratifiedSampler
from RapidECalibrationDataset import RapidECalibrationDataset
from Standardize import RapidECalibrationStandardize
from RapidEClassifier import RapidEClassifier
from torch.utils.data import DataLoader
from StratifiedSplitter import StratifiedSplitterForRapidECalibDataset
from torch.utils.tensorboard import SummaryWriter
from NestedCrossValidator import NestedCrossValidator


def main(argv):

    with open(argv[0],'r') as f:
        hyperparameters = json.load(f)

    dataset_meta_json_path = hyperparameters['dataset_meta_json_path']
    writer = SummaryWriter(log_dir='../runs/' + hyperparameters['experiment_name'], flush_secs=10)

    standardization = RapidECalibrationStandardize(dataset_meta_json_path)
    dataset = RapidECalibrationDataset(dataset_meta_json_path, transform=standardization)

    model = RapidEClassifier(number_of_classes=hyperparameters['num_of_classes'], dropout_rate=hyperparameters['dropout_rate'], name=hyperparameters['experiment_name'])


    # SET GPU
    if hyperparameters['gpu']:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = torch.nn.DataParallel(model)
        else:
            device = torch.device("cpu")
            model = model
            logging.warning('CUDA is not availible on this machine')
    else:
        device = torch.device("cpu")
        model = model
    model.to(device)

    # OBJECTIVE LOSS
    labels = dataset.gettargets()
    set_l = set(labels)
    freq = stats.relfreq(np.array(labels), numbins=len(set_l)).frequency
    weights = torch.tensor((1/freq)/np.sum(1/freq), dtype=torch.float32)
    weights = weights.to(device)
    loss = nn.CrossEntropyLoss(weight=weights, reduction='sum')

    splitter = StratifiedSplitterForRapidECalibDataset(hyperparameters['num_of_folds'],dataset)

    # NESTED CROSS-VALIDATION
    nestedcrossvalidator =NestedCrossValidator(model=model, device=device, objectiveloss=loss,splitter=splitter, hyperparams=hyperparameters, tbwriter=writer)
    nestedcrossvalidator()
    writer.close()


if __name__ == "__main__":
   main(sys.argv[1:])