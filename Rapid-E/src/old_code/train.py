import torch
import numpy as np
from dataset import RapidECalibrationDataset, Standardize
from torch import nn
from torch.utils.data import DataLoader
from sampler import StratifiedSampler
from scipy import stats
from model import RapidENetClassifier
import torch.optim as optim
import logging
import optuna
import sys
import os
import random
from sklearn.metrics import accuracy_score

from random import sample


base_dir = '/home/guest/coderepos/transfer_learning/Rapid-E'
tensor_subdir = 'data/calib_data_ns_tensors'
meta_subdir = 'meta_jsons'
dataset_meta_json_path = '/home/guest/coderepos/transfer_learning/Rapid-E/meta_jsons/00000_train.pt'
num_out_folds = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_of_epochs = 5





dd = torch.load(dataset_meta_json_path)
labels = dd['data'][2]
set_l = set(labels)
freq = stats.relfreq(np.array(labels), numbins=len(set_l)).frequency
weights = torch.Tensor((1/freq)/np.sum(1/freq))
weights = weights.to(device)
loss = nn.CrossEntropyLoss(weight=weights, reduction='sum')





#dataset_meta_json = torch.load(dataset_meta_json_path)
#splits = sample(list(range(len(dataset_meta_json['train_splits']))), num_out_folds)
#splits.sort()
#splits = [(dataset_meta_json['train_splits'][i],torch.load(os.path.join(base_dir,meta_subdir,dataset_meta_json['train_splits'][i]))['pair_split_path']) for i in range(num_out_folds)]








def objective(train_dataset, valid_dataset=None, trial = None, best_params = None):
    labels = train_dataset.dd['data'][2]
    sampler = StratifiedSampler(torch.Tensor(labels), 20800)
    train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=20800, num_workers=50)
    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset,batch_size=5200, num_workers=50)
    if best_params is None:
        alpha = trial.suggest_loguniform("alpha", 1e-5, 1e-1)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 1.0)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 1.0)
    else:
        alpha = best_params['alpha']
        dropout_rate = best_params['dropout_rate']
        weight_decay = best_params['weight_decay']


    model = RapidENetClassifier(number_of_classes=26, dropout_rate=dropout_rate)

    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      model = nn.DataParallel(model)

    model.to(device)
    optimizerAdam = optim.Adam(model.parameters(), lr = alpha, weight_decay=weight_decay)
    schedulerAdam = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerAdam, patience=4, factor=0.5, verbose=False)
    # if valid_dataset is not None:
    #     valid_total_loss = 1e+10
    for epoch in range(num_of_epochs):
        running_loss = 0
        for idx, data in enumerate(train_loader):
            optimizerAdam.zero_grad()
            for key in data.keys():
                if key == 'Timestamp':
                    continue
                data[key] = data[key].to(device)
            out=model(data) # forward
            loss_value = loss(out,data['Label'])
            loss_value.backward()
            optimizerAdam.step()
            schedulerAdam.step(loss_value)
            running_loss += loss_value.item()
            #if idx % 1000 ==0:
            #    print('\tBatch ' + str(idx) + ' loss:', loss_value.item())
        print('Epoch ' + str(epoch) + ' train loss: ', running_loss)

        if valid_dataset is not None:
            valid_total_loss = 0
            for data in valid_loader:
                for key in data.keys():
                    if key == 'Timestamp':
                        continue
                    data[key] = data[key].to(device)
                out_val = model(data)
                loss_valid = loss(out_val,data['Label'])
                valid_total_loss += loss_valid.item()
            print('Epoch ' + str(epoch) + ' valid loss: ', valid_total_loss)
            if trial is not None:
                intermediate_value = valid_total_loss
                trial.report(intermediate_value, epoch)

                if trial.should_prune():
                    raise optuna.TrialPruned()



    if trial is None:
        if valid_dataset is None:
            return running_loss, model
        else:
            return valid_total_loss, model
    else:
        if valid_dataset is None:
            return running_loss
        else:
            return valid_total_loss




def cross_validation(dataset_meta_path):
    #create_split
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner())

    dataset_meta = torch.load(dataset_meta_path)
    splits = sample(list(range(len(dataset_meta['train_splits']))), num_out_folds)
    splits.sort()
    splits = [(dataset_meta['train_splits'][i],
               torch.load(os.path.join(base_dir, meta_subdir, dataset_meta['train_splits'][i]))['pair_split_path'])
              for i in range(num_out_folds)]

    # s obzirom na trenutni split definirati cross-validacijski objective
    def objective_cv(trail):
        losses = []
        for train, valid in splits:
            train_path = os.path.join(base_dir, meta_subdir, train)
            standardization = Standardize(train_path)
            if valid is not None:
                valid_path = os.path.join(base_dir, meta_subdir, valid)
                valid_dataset = RapidECalibrationDataset(valid_path, transform=standardization)
            else:
                valid_dataset = None


            train_dataset = RapidECalibrationDataset(train_path, transform=standardization)

            loss_v = objective(trial=trail, train_dataset=train_dataset, valid_dataset=valid_dataset)
            losses.append(loss_v)
        obj = np.mean(np.array(losses))
        #print(obj)
        return obj

    standardization = Standardize(dataset_meta_path)
    dataset = RapidECalibrationDataset(dataset_meta_path, transform=standardization)
    study.optimize(objective_cv, n_trials=2)
    final_loss, final_model = objective(train_dataset=dataset, best_params=study.best_params)
    return final_loss, final_model


def predict_from_prob(probs):
    pmax = torch.max(probs)
    classes = []
    for i, p in enumerate(probs):
        if p == pmax:
            classes.append(i)

    if len(classes) == 1:
        return classes[0]
    else:
        return classes[random.randrange(0,len(classes))]

def nested_cross_validation(dataset_meta_path):
    #optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    #study = optuna.create_study(pruner=optuna.pruners.MedianPruner())

    dataset_meta = torch.load(dataset_meta_path)
    splits = sample(list(range(len(dataset_meta['train_splits']))), num_out_folds)
    splits.sort()
    splits = [(dataset_meta['train_splits'][i],
               torch.load(os.path.join(base_dir, meta_subdir, dataset_meta['train_splits'][i]))['pair_split_path'])
              for i in range(num_out_folds)]
    losses = []
    accuracies = []
    for trainvalid, test in splits:
        _ , cross_valid_model = cross_validation(os.path.join(base_dir,meta_subdir,trainvalid))
        trainvalid_path = os.path.join(base_dir,meta_subdir,trainvalid)
        test_path = os.path.join(base_dir,meta_subdir,test)
        standardization = Standardize(trainvalid_path)
        test_dataset = RapidECalibrationDataset(test_path, transform=standardization)
        test_loader = DataLoader(test_dataset, batch_size=5200, num_workers=50)

        # calculate loss on the test set
        cross_valid_loss = 0
        #predicted_values = []
        predicted_labels = []
        for data in test_loader:
            for key in data.keys():
                if key == 'Timestamp':
                    continue
                data[key] = data[key].to(device)
            out_val = cross_valid_model(data)

            loss_valid = loss(out_val, data['Label'])
            predicted_labels += [predict_from_prob(out_val[i]) for i in range(len(out_val))]

            #print(out_val.shape)
            cross_valid_loss += loss_valid.item()
        real_labels = torch.load(test_path)['data'][2]
        cross_valid_accuracy = accuracy_score(real_labels, predicted_labels)

        accuracies.append(cross_valid_accuracy)
        losses.append(cross_valid_loss)
    print('Mean loss: ', np.mean(losses))
    print('Mean accuracy: ', np.mean(accuracies))

    # retraining of the model
    cross_valid_loss, cross_valid_model = cross_validation(dataset_meta_path)
    return cross_valid_loss, cross_valid_model


ffloss, trained_model = nested_cross_validation(dataset_meta_json_path)


