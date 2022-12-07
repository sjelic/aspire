import torch
import optuna
import logging
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Predictor import Predictor
import numpy as np
from torch.nn import DataParallel

def get_model(model):
    if isinstance(model,DataParallel):
        return model.module
    else:
        return model


class Trainer:
    def __init__(self, model, device, objectiveloss, trainloader, validloader=None, hyperparams=None, tbwriter = None):
        # model - must be the instance of nn.Module
        # train_dataset - must be an instance of Dataet
        # valid_dataset - must be an instance of Dataset
        # hyperparams - dictionary of hyperparams:
        # optuna Trial object
        #if isinstance(model, torch.nn.Module):
        #    self.model = model
        #else:
        #    logging.error('Model must be an instance of torch.nn.Module')
        self.model = model
        self.device = device
        if isinstance(objectiveloss, Module):
            self.objective_loss = objectiveloss
        else:
            logging.error('ObjectiveLoss must be an instance of torch.nn.Module')

        if isinstance(trainloader, DataLoader):
            self.train_loader = trainloader
        else:
            logging.error('TrainDataset must be an instance of torch.nn.Dataloader')

        if (validloader is None) or isinstance(validloader, DataLoader):
            self.valid_loader = validloader
        else:
            logging.error('ValidDataset must be an instance of torch.nn.Dataloader')

        self.nEp = hyperparams['num_of_epochs'] if hyperparams is not None and type(
            hyperparams) == dict and 'num_of_epochs' in hyperparams.keys() else 10
        self.lRa = hyperparams['learning_rate'] if hyperparams is not None and type(
            hyperparams) == dict and 'learning_rate' in hyperparams.keys() else 0.1
        self.wDe = hyperparams['weight_decay'] if hyperparams is not None and type(
            hyperparams) == dict and 'weight_decay' in hyperparams.keys() else 5e-4
        # self.GPU = hyperparams['gpu'] if hyperparams is not None and type(
        #     hyperparams) == dict and 'gpu' in hyperparams.keys() else False
        # self.nGp = hyperparams['num_of_gpus'] if hyperparams is not None and type(
        #     hyperparams) == dict and 'num_of_gpus' in hyperparams.keys() else 1

        #self.writer = tbwriter

        if tbwriter is not None:
            if validloader is not None:
                self.writer = SummaryWriter(log_dir=tbwriter.log_dir + '/hp_tune/' + trainloader.dataset.name)
            else:
                self.writer = SummaryWriter(log_dir=tbwriter.log_dir + '/' + trainloader.dataset.name)





        # all columns
        if hyperparams is not None and type(hyperparams) == dict and hyperparams['optimizer'] == 'Adam':
            self.oPt = torch.optim.Adam(self.model.parameters(), lr=self.lRa, weight_decay=self.wDe)
            self.sHd = torch.optim.lr_scheduler.ReduceLROnPlateau(self.oPt, patience=5, factor=0.8, verbose=False)

        elif hyperparams is not None and type(hyperparams) == dict and hyperparams['optimizer'] == 'SGD':
            self.oPt = torch.optim.SGD(self.model.parameters(), lr=self.lRa, weight_decay=self.wDe)
            self.sHd = torch.optim.lr_scheduler.ReduceLROnPlateau(self.oPt, patience=5, factor=0.8, verbose=False)

        elif hyperparams is None:
            self.oPt = torch.optim.Adam(self.model.parameters(), lr=self.lRa, weight_decay=self.wDe)
            self.sHd = torch.optim.lr_scheduler.ReduceLROnPlateau(self.oPt, patience=5, factor=0.8, verbose=False)

        else:
            logging.error('Optimizer unsupported.')

        self.predictor = Predictor(model=self.model, hyperparams=hyperparams)

    def transfer_to_device(self, data):
        for key in get_model(self.model).features:
            data[key] = data[key].to(self.device)
        data['Target'] = data['Target'].to(self.device)

    def init_layers(self, model):
        if type(model) == torch.nn.Linear or type(model) == torch.nn.Conv2d or type(model) == torch.nn.Conv1d:
            torch.nn.init.xavier_uniform_(model.weight)
            model.bias.data.fill_(0.01)

    def init_weights(self, model):
        model.apply(self.init_layers)


    def __call__(self, trial=None):
        self.init_weights(self.model)

        if trial is not None:
            if isinstance(trial, optuna.trial.Trial):
                self.lRa = trial.suggest_loguniform("alpha", 1e-5, 1e-1)
                get_model(self.model).dropout_rate = trial.suggest_float("dropout_rate", 0.0, 1.0)
                self.wDe = trial.suggest_float("weight_decay", 0.0, 1.0)
                self.nEp = 20
            else:
                logging.error('HyperOptTrial must be an instance of optuna.trial.Trial')

        for epoch in range(self.nEp):
            running_loss = 0
            running_acc = 0
            tb_dict = {}
            ac_dict = {}
            best_accuracy = 0
            best_loss = np.inf
            for idx, data in enumerate(self.train_loader):
                self.transfer_to_device(data)
                out = self.model(data).to(self.device)
                loss_value = self.objective_loss(out, data['Target'])
                self.oPt.zero_grad()
                loss_value.backward()
                self.oPt.step()
                self.sHd.step(loss_value)

            #running_loss = 0
            #train_predictions = torch.tensor([], dtype=torch.int64).to(self.device)
            #train_targets = torch.tensor([], dtype=torch.int64).to(self.device)
            for idx, data in enumerate(self.train_loader):
                self.transfer_to_device(data)
                out = self.model(data).to(self.device)
                preds = torch.argmax(out, dim=1).to(self.device)
                running_acc += torch.sum(torch.eq(preds, data['Target'])).item()
                running_loss += self.objective_loss(out, data['Target']).item()

                #train_predictions = torch.cat([train_predictions, preds], dim=0)
                #train_targets = torch.cat([train_targets, data['Target']], dim=0)

            tb_dict['train_loss'] = running_loss / len(self.train_loader.dataset)
            ac_dict['train_accuracy'] = 1.0 * running_acc / len(self.train_loader.dataset)
            #print('Epoch %d train loss: %f' % (epoch + 1, running_loss / len(self.train_loader.dataset)))

            if self.valid_loader is not None:
                valid_acc = 0
                valid_total_loss = 0
                #val_predictions = torch.tensor([], dtype=torch.int64).to(self.device)
                #val_targets = torch.tensor([], dtype=torch.int64).to(self.device)
                for data in self.valid_loader:
                    self.transfer_to_device(data)
                    out_val = self.model(data).to(self.device)
                    preds_val = torch.argmax(out_val, dim=1).to(self.device)
                    #val_predictions = torch.cat([val_predictions,preds_val], dim=0)
                    #preds_val = torch.tensor(self.predictor(out_val),dtype=torch.int64).to(self.device)
                    valid_acc += torch.sum(torch.eq(preds_val,data['Target'])).item()
                    #val_targets = torch.cat([val_targets, data['Target']], dim=0)

                    loss_valid = self.objective_loss(out_val, data['Target'])
                    valid_total_loss += loss_valid.item()

                if trial is not None:
                    intermediate_value = valid_total_loss
                    trial.report(intermediate_value, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                tb_dict['valid_loss'] = valid_total_loss / len(self.valid_loader.dataset)
                ac_dict['valid_accuracy'] = 1.0*valid_acc / len(self.valid_loader.dataset)
                if best_accuracy < ac_dict['valid_accuracy']:
                    torch.save(self.model, 'best_model.pt')
                    best_accuracy = ac_dict['valid_accuracy']
                    best_loss = tb_dict['valid_loss']
                #print('Epoch %d valid loss: %f' % (epoch + 1, valid_total_loss / len(self.valid_loader.dataset)))
                #self.writer.add_pr_curve('PR-Curve on Valid Set', val_targets, val_predictions, epoch+1)
            else:
                if best_accuracy < ac_dict['train_accuracy']:
                    torch.save(self.model, 'best_model.pt')
                    best_accuracy = ac_dict['train_accuracy']
                    best_loss = tb_dict['train_loss']
                #self.writer.add_pr_curve('PR-Curve on Train Set', train_targets, train_predictions, epoch+1)

            if self.writer is not None:
                self.writer.add_scalars('loss', tb_dict, epoch + 1)
                self.writer.add_scalars('accuracy', ac_dict, epoch + 1)

            if self.oPt.param_groups[0]['lr'] < 1e-10:
                print('Learning rate is too small (<1e-10). Training process is interrupted.')
                break;




        #self.writer.close()
        self.model = torch.load('best_model.pt')
        #get_model(self.model).load_state_dict(stdict)


        return best_loss

