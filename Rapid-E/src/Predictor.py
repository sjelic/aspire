import torch
import logging
import random
from torch.nn import  DataParallel


def get_model(model):
    if isinstance(model,DataParallel):
        return model.module
    else:
        return model


class Predictor:
    def __init__(self, model, hyperparams = None):
        self.model = model
        self.hyperparams = hyperparams
        # self.GPU = hyperparams['gpu'] if hyperparams is not None and type(hyperparams) == dict and 'gpu' in hyperparams.keys() else False
        # if self.GPU:
        #     if torch.cuda.is_available():
        #         self.device = torch.device("cuda")
        #         if ~isinstance(self.model,torch.nn.DataParallel):
        #             self.features = self.model.module.features
        #             self.model = torch.nn.DataParallel(self.model)
        #         else:
        #             self.features = self.model.module.features
        #     else:
        #         self.features = self.model.features
        #         self.device = torch.device("cpu")
        #         logging.warning('CUDA is not availible on this machine')
        #
        #     self.model.to(self.device)
    # put tensor
    def predict_labels(self, probs):

        if get_model(self.model).type == 'Classifier':
            return torch.argmax(probs, dim=1)
        else:
            return probs


        # pmax = torch.max(sample)
        # classes = []
        # for i, p in enumerate(sample):
        #     if p == pmax:
        #         classes.append(i)
        #
        # if len(classes) == 1:
        #     return classes[0]
        # else:
        #     return classes[random.randrange(0, len(classes))]

    def compute_probs(self, input_loader):
        model_out = []
        for data in input_loader:
            #for key in get_model(self.model).features:
            #    data[key] = data[key].to(self.device)
            model_out += self.model(data)
        return model_out

    # def compute_loss(self, probs, objective_loss):
    #     loss_out = 0
    #     for data in input_loader:
    #         for key in self.model.features:
    #             data[key] = data[key].to(self.device)
    #         data['Target'] = data['Target'].to(self.device)
    #         model_out = self.model(data)
    #     return objective_loss(model_out, data['Target']).item()
    #
    #     return loss_out

    def __call__(self, probs):
        return [self.predict_labels(probs[i]) if get_model(self.model).type == 'Classifier' else probs[i] for i in
                      range(len(probs))]
