import torch
import logging
import numpy as np
from CrossValidator import CrossValidator
from Predictor import Predictor
from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score, explained_variance_score
from Sampler import StratifiedSampler
from torch.utils.data import DataLoader
from StratifiedSplitter import StratifiedSplitterForRapidECalibDataset
from Trainer import get_model
from confusion_matrix import plot_confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import PIL.Image
from torchvision.transforms import ToTensor

class NestedCrossValidator:
    def __init__(self, model, device, objectiveloss, splitter, hyperparams = None, tbwriter = None):

        if(isinstance(model, torch.nn.Module)):
            self.model = model
        else:
            logging.error('model must be an instance of torch.nn.Module class')


        if isinstance(objectiveloss, torch.nn.Module):
            self.objectiveloss = objectiveloss
        else:
            logging.error('ObjectiveLoss must be an instance of torch.nn.Module')

        self.device = device
        self.splitter = splitter
        self.hyperparams = hyperparams
        self.tbSz = hyperparams['train_batch_size'] if hyperparams is not None and type(
            hyperparams) == dict and 'train_batch_size' in hyperparams.keys() else 10
        self.vbSz = hyperparams['valid_batch_size'] if hyperparams is not None and type(
            hyperparams) == dict and 'valid_batch_size' in hyperparams.keys() else 2
        self.nWo = hyperparams['num_workers'] if hyperparams is not None and type(
            hyperparams) == dict and 'num_workers' in hyperparams.keys() else 1
        self.nTr = hyperparams['num_trials'] if hyperparams is not None and type(
            hyperparams) == dict and 'num_trials' in hyperparams.keys() else 10
        #self.metrics = hyperparams['metrics'] if hyperparams is not None and type(
        #    hyperparams) == dict and 'metrics' in hyperparams.keys() else ['Loss']
        self.num_of_classes = get_model(self.model).number_of_classes
        self.evaluation = {}
        self.evaluation['Accuracy'] = []
        self.evaluation['Precision'] = []
        self.evaluation['Recall'] = []
        self.evaluation['Objective Loss'] = []
        #for metr in self.metrics:
        #    self.evaluation[metr] = []
        #self.evaluation['ObjectiveLoss'] = []

        if tbwriter is not None:
            self.writer = SummaryWriter(log_dir=tbwriter.log_dir + '/outer_cv_'+ self.splitter.dataset.name)

        #self.final_loss = np.inf
    def __call__(self):
        num_of_classes = get_model(self.model).number_of_classes
        confusion_train_matrix_final = torch.zeros((num_of_classes, num_of_classes),
                                             dtype=torch.float64).to(self.device)
        confusion_test_matrix_final = torch.zeros((num_of_classes, num_of_classes),
                                                   dtype=torch.float64).to(self.device)

        for train, test in self.splitter():

            train_sampler = StratifiedSampler(torch.tensor(train.gettargets()), batch_size=self.tbSz)
            test_sampler = StratifiedSampler(torch.tensor(test.gettargets()), batch_size=self.vbSz)
            train_loader = DataLoader(train, sampler=train_sampler, batch_size=self.tbSz, num_workers=self.nWo)
            test_loader = DataLoader(test, sampler=test_sampler, batch_size=self.vbSz, num_workers=self.nWo)
            #self.splitter.dataset = train
            splitter = StratifiedSplitterForRapidECalibDataset(self.hyperparams['num_of_folds'], train)

            crossvalidator = CrossValidator(model=self.model,device=self.device, objectiveloss=self.objectiveloss, splitter=splitter,hyperparams=self.hyperparams, tbwriter=self.writer)
            crossvalidator()


            confusion_train_matrix = torch.zeros((num_of_classes, num_of_classes), dtype=torch.float64).to(self.device)
            for idx, data in enumerate(train_loader):
                for key in get_model(self.model).features:
                    data[key] = data[key].to(self.device)
                data['Target'] = data['Target'].to(self.device) # real
                out = self.model(data).to(self.device)
                preds = torch.argmax(out, dim=1).to(self.device) # predictions
                #print(preds.shape[0])
                for j in range(preds.shape[0]):
                    confusion_train_matrix[data['Target'][j], preds[j]] += 1

                #running_acc += torch.sum(torch.eq(preds, data['Target'])).item()
                #running_loss += self.objective_loss(out, data['Target']).item()

            confusion_train_matrix_final += confusion_train_matrix
            confusion_test_matrix = torch.zeros((num_of_classes, num_of_classes),
                                                 dtype=torch.float64).to(self.device)
            for idx, data in enumerate(test_loader):
                for key in get_model(self.model).features:
                    data[key] = data[key].to(self.device)
                data['Target'] = data['Target'].to(self.device)
                out = self.model(data).to(self.device)
                preds = torch.argmax(out, dim=1).to(self.device)
                #print(preds.shape[0])
                for j in range(preds.shape[0]):
                    confusion_test_matrix[data['Target'][j], preds[j]] += 1
                #running_acc += torch.sum(torch.eq(preds, data['Target'])).item()
                #running_loss += self.objective_loss(out, data['Target']).item()
            confusion_test_matrix_final += confusion_test_matrix
            confusion_test_matrix_row_sum = torch.sum(confusion_test_matrix,dim=1)
            confusion_test_matrix_column_sum = torch.sum(confusion_test_matrix, dim=0)

            self.evaluation['Precision'].append([confusion_test_matrix[i][i] / confusion_test_matrix_column_sum[i] for i in range(num_of_classes)])
            self.evaluation['Recall'].append([confusion_test_matrix[i][i]/confusion_test_matrix_row_sum[i] for i in range(num_of_classes)])

            accuracy = []


            plot_buf_train = plot_confusion_matrix(cm=np.array(confusion_train_matrix.cpu()),
                                                   target_names=list(range(num_of_classes)))
            plot_buf_test = plot_confusion_matrix(cm=np.array(confusion_test_matrix.cpu()),
                                                  target_names=list(range(num_of_classes)))
            image_train = PIL.Image.open(plot_buf_train)
            image_train = ToTensor()(image_train).unsqueeze(0)
            image_test = PIL.Image.open(plot_buf_test)
            image_test = ToTensor()(image_test).unsqueeze(0)
            self.writer.add_image('train_confusion_matrix', image_train, global_step=idx + 1, dataformats='NCHW')
            self.writer.add_image('test_confusion_matrix', image_test, global_step=idx + 1,  dataformats='NCHW')

        plot_buf_train = plot_confusion_matrix(cm = np.array(confusion_train_matrix_final.cpu()), target_names=list(range(num_of_classes)))
        plot_buf_test = plot_confusion_matrix(cm=np.array(confusion_test_matrix_final.cpu()),                                             target_names=list(range(num_of_classes)))
        image_train = PIL.Image.open(plot_buf_train)
        image_train = ToTensor()(image_train).unsqueeze(0)
        image_test = PIL.Image.open(plot_buf_test)
        image_test= ToTensor()(image_test).unsqueeze(0)
        self.writer.add_image('train_confusion_matrix_final', image_train, dataformats='NCHW')
        self.writer.add_image('test_confusion_matrix_final', image_test, dataformats='NCHW')





