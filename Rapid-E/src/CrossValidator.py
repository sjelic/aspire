from Trainer import Trainer
import logging
import optuna
import torch
import numpy as np
from torch.nn import Module
from torch.utils.data import DataLoader
from Sampler import StratifiedSampler
from torch.utils.tensorboard import SummaryWriter


class CrossValidator:

    def __init__(self, model, device, objectiveloss, splitter, hyperparams = None, tbwriter = None):
        if (isinstance(model, Module)):
            self.model = model
        else:
            logging.error('model must be an instance of Module class')

        if isinstance(objectiveloss, Module):
            self.objectiveloss = objectiveloss
        else:
            logging.error('ObjectiveLoss must be an instance of Module class')

        self.device = device
        self.splitter = splitter
        self.hyperparams = hyperparams
        self.tbSz = hyperparams['train_batch_size'] if hyperparams is not None and type(hyperparams) == dict and 'train_batch_size' in hyperparams.keys() else 10
        self.vbSz = hyperparams['valid_batch_size'] if hyperparams is not None and type(hyperparams) == dict and 'valid_batch_size' in hyperparams.keys() else 2
        self.nWo  = hyperparams['num_workers'] if hyperparams is not None and type(hyperparams) == dict and 'num_workers' in hyperparams.keys() else 1
        self.nTr  = hyperparams['num_trials'] if hyperparams is not None and type(hyperparams) == dict and 'num_trials' in hyperparams.keys() else 10
        self.final_loss = np.inf
        #optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        self.study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
        #self.writer = tbwriter
        if tbwriter is not None:
            self.writer = SummaryWriter(log_dir=tbwriter.log_dir + '/cv_'+ self.splitter.dataset.name)


    def cross_valid_objective(self, trial):

        losses = []
        for train, valid in self.splitter():
            train_sampler = StratifiedSampler(torch.tensor(train.gettargets()),batch_size=self.tbSz)
            valid_sampler = StratifiedSampler(torch.tensor(valid.gettargets()),batch_size=self.vbSz)
            train_loader = DataLoader(train, sampler=train_sampler, batch_size=self.tbSz, num_workers=self.nWo)
            valid_loader = DataLoader(valid, sampler=valid_sampler, batch_size=self.vbSz, num_workers=self.nWo)
            trainer = Trainer(self.model, self.device, self.objectiveloss, trainloader=train_loader, validloader=valid_loader, hyperparams=self.hyperparams, tbwriter=self.writer)
            loss_v = trainer(trial=trial)
            losses.append(loss_v)
        return np.mean(losses)



    def __call__(self):

        self.study.optimize(self.cross_valid_objective, n_trials=self.nTr)

        if self.hyperparams is None:
            self.hyperparams = {}

        for key in self.study.best_params:
            self.hyperparams[key] = self.study.best_params[key]

        train_sampler = StratifiedSampler(torch.tensor(self.splitter.dataset.gettargets()), batch_size=self.tbSz)
        train_loader = DataLoader(self.splitter.dataset, sampler=train_sampler, batch_size=self.tbSz,num_workers=self.nWo)
        final_trainer = Trainer(self.model, self.device, self.objectiveloss, trainloader=train_loader, validloader=None, hyperparams=self.hyperparams, tbwriter=self.writer)
        self.final_loss = final_trainer()



