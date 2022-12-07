# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 00:46:06 2020

@author: sjelic
"""
import sys
import os
import copy
#import pandas as pd
import numpy as np
import torch
import logging
import pandas as pd
from torch import nn
#from torch import nn
#import matplotlib.pyplot as plt
#import functools as fnc
from torch.utils.data import DataLoader
from sampler import StratifiedSampler
from utils import train_test_split, my_collate
import torch.optim as optim
from objectives import WeightedSELoss, PearsonCorrelationLoss
from dataset import RapidEDataset
from model import RapidENetCUDA
#from torch.nn.parallel import DistributedDataParallel as DDP

#from parallel import DataParallelModel, DataParallelCriterion
args = {
'experiment_name': 'Experiment',
'data_dir_path': '../data/novi_sad_2019_/',
'model_dir_path': './models',
'metadata_path': './Libraries/probe_dataset.xlsx',
'pollen_info_path': './Libraries/pollen_types.xlsx',
'hparam_path': 'hyper_params.json',
'objective_criteria': 'WeightedSELoss',
'additional_criteria': ['PearsonCorrelationLoss'],
'selection_criteria': 'WeightedSELoss',
'cross_valid_type': 'cross_seasonal',
'hparam_search_strategy': 'gridsearch',
'num_of_valid_splits': 2,
'num_of_test_splits': 2,
'train_batch_size': 10,
'valid_batch_size': 5,
'test_batch_size': 5,
'model': 'RapidENet',
'pretrained_model_state_path': './models/novi_sad/model_pollen_types_ver0/Ambrosia_vs_all.pth',
'number_of_classes': 2,
'logging_per_batch': True,
'logging': True,
'load_entire_dataset' : False,
'entire_set_pickle': '../data/novi_sad_2019.pkl',
'GPU': True
}



        
class Experiment:
    def __init__(self):
        pass

    def prepare_data(self):
        pass

    def train_model(self):
        pass

    def validate_model(self):
        self

    def deploy_model(self):
        self

    def predict(self):

 
    
    
    
    def set_model(self, hp, obj_criteria, model_path = None):
        
        if self.args['model'] == 'RapidENet':
            self.model = RapidENetCUDA(obj_loss=obj_criteria, dropout_rate = hp['drop_out'], number_of_classes = self.args['number_of_classes']).float()
            if model_path:
                self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
            else:
                self.model.load_state_dict(torch.load(self.args['pretrained_model_state_path'], map_location=lambda storage, loc: storage), strict=False)
            if self.args['GPU']:
                self.model = nn.DataParallel(self.model)
                self.model.to(self.device)

        else:
            raise RuntimeError('Only RapidENet is implemented.')
        
        
        
    def set_optimizer(self, hp):
        if hp['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr = hp['lr'], weight_decay=hp['weight_decay'])
            #self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=hp['lr']/10, max_lr=hp['lr'])
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=4, factor=0.5, verbose=False)
            
        elif hp['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr = hp['lr'], momentum=hp['momentum'], weight_decay=hp['weight_decay'], nesterov=True)
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=hp['lr']/10, max_lr=hp['lr'])
        
        elif hp['optimizer'] == 'lbfgs':
            self.optimizer = optim.LBFGS(self.model.parameters(), lr =  hp['lr'], max_iter=hp['max_iter'], line_search_fn='strong_wolfe')
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=hp['lr']/10, max_lr=hp['lr'])
        
        else:
            logging.warning('This optimizer is not implemented. Adam will be used instead.')
            self.optimizer = optim.AdamW(self.model.parameters(), lr = hp['lr'], weight_decay=hp['weight_decay'])
        
    
    def set_metric_by_name(self, name):
        if name ==  'WeightedSELoss':
            criteria = WeightedSELoss(selection=(True if name == self.args['selection_criteria'] else False))
            if self.args['GPU']:
                #criteria = DataParallelCriterion(criteria)
                criteria = nn.DataParallel(criteria)
                criteria.to(self.device)
                #criteria = criteria.cuda(0)
        if name ==  'PearsonCorrelationLoss':
            criteria = PearsonCorrelationLoss(selection=(True if name == self.args['selection_criteria'] else False))
            if self.args['GPU']:
                criteria = nn.DataParallel(criteria)
                criteria.to(self.device)
                #criteria = criteria.cuda(0)
                #criteria = criteria.to(self.device)
                #criteria = nn.DataParallel(criteria)


       
        
        return criteria
    
    def set_train_dict(self, num_of_epochs):
        self.train_dict = {'train': {},
                            'valid': {}}
        
        
        for dset in ['train', 'valid']:
            for criteria in [self.criteria['objective_criteria']] + self.criteria['additional_criteria']:
                # name = (criteria.module.name if self.args['GPU'] else criteria.name)
                # sense = (criteria.module.sense if self.args['GPU'] else criteria.sense)

                #name = criteria.name
                #sense = criteria.sense
                name = 'WeightedSELoss'
                sense = 'min'
                self.train_dict[dset][name] = {'epochs_sum': torch.zeros(num_of_epochs),
                                                            'epochs_mean': torch.zeros(num_of_epochs),
                                                            'best_value': float('inf') if  sense == 'min' else float('-inf'),
                                                             }
    
    def prepare_data_loader(self, dframe, batch_size, dataset_name):
        dataset = RapidEDataset(dframe, self.args['data_dir_path'], self.df_pollen_types, load=self.args['load_entire_dataset'], name = dataset_name, preloaded_dict=self.preloaded_dict)
        stratified_train_sampler = StratifiedSampler(torch.from_numpy(np.array(list(dframe['CLUSTER']))), batch_size)
        return DataLoader(dataset, batch_size=batch_size, sampler = stratified_train_sampler, collate_fn=my_collate, pin_memory=True)

    
    def tune_hparam(self, inner):
        means = torch.mean(inner,1)
        #stdevs = torch.std(inner,1)
        #sense = (self.criteria['selection_criteria'].module.sense if self.args['GPU'] else self.criteria['selection_criteria'].sense)
        sense = self.criteria['selection_criteria'].sense

        if ( sense == 'max'):
            jopt = torch.argmax(means).item()
        else:
            jopt = torch.argmin(means).item()
        return jopt
    
    

            
    def update_batch_info(self, dataset_type, output, target, weights, batch_idx, epoch_idx, num_of_batches):
        for crt in [self.criteria['objective_criteria']] + self.criteria['additional_criteria']:
            #name = (crt.module.name if self.args['GPU'] else crt.name)
            #batch_loss = crt(output,target,weights)
            name = crt.name
            batch_loss = crt(output,target,weights)
            self.train_dict[dataset_type][name]['epochs_sum'][epoch_idx] += batch_loss.item()
            if (batch_idx == num_of_batches - 1):
                self.train_dict[dataset_type][name]['epochs_mean'][epoch_idx] = self.train_dict[dataset_type][name]['epochs_sum'][epoch_idx] / num_of_batches

    def nested_crossvalidation(self):
        if self.logging:
            print('Nested crossvalidation started.')
            print('Number of train_valid-test splits: ' + str(self.args['num_of_test_splits']))
            print('Number of train-valid splits:' + str(self.args['num_of_valid_splits']))
        
        test_split_groups = np.array(list(self.df['CLUSTER']))
        train_valid_test = train_test_split(test_split_groups, num_splits = self.args['num_of_test_splits'])
        outer = torch.zeros(self.args['num_of_test_splits'])
        for i, (train_valid_data, test_data) in enumerate(train_valid_test):
            # print('Fold ' + str(i+1))
            # print('Train dataset: ' + str(len(train_valid_data)))
            # print('Test dataset: ' + str(len(test_data)))
            # train model
            
            df_train_valid = self.df.iloc[sorted(train_valid_data)]
            #print(df_train_valid.index.tolist())
            df_train_valid = df_train_valid.set_index(pd.Index(list(range(len(df_train_valid)))))
            #print(df_train_valid.index.tolist())
            #break
            valid_split_groups = np.array(list(df_train_valid['CLUSTER']))
            train_valid = train_test_split(valid_split_groups, num_splits = self.args['num_of_valid_splits'])
            inner = torch.zeros(len(self.hyperparameters), self.args['num_of_valid_splits'])
            for j, (train_data, valid_data) in enumerate(train_valid):
                # print(j)
                # print(len(train_data))
                # print(len(valid_data))
                
                df_train = df_train_valid.iloc[sorted(train_data)]
                df_train = df_train.set_index(pd.Index(list(range(len(df_train)))))
                train_loader = self.prepare_data_loader(df_train, self.args['train_batch_size'], str(i+1) + '_' + str(j+1)+ '_' +'train')
                df_valid = df_train_valid.iloc[sorted(valid_data)]
                df_valid = df_valid.set_index(pd.Index(list(range(len(df_valid)))))
                valid_loader = self.prepare_data_loader(df_valid, self.args['valid_batch_size'], str(i+1) + '_' + str(j+1)+ '_' +'valid')
                
                for k, hp in enumerate(self.hyperparameters):
                    
                    inner[k][j] = self.train(train_loader, hp, save_model = False, valid_loader=valid_loader)

            hp_best = self.hyperparameters[self.tune_hparam(inner)]
            df_test = self.df.iloc[sorted(test_data)]
            trainvalid_loader = self.prepare_data_loader(train_valid, self.args['train_batch_size'], str(i+1) + '_' +'trainvalid')
            test_loader = self.prepare_data_loader(df_test, self.args['test_batch_size'], str(i+1) + '_' +'test')
            outer[i] = self.train(trainvalid_loader, hp_best, save_model=True, valid_loader=test_loader)
            
        return {'mean_objective_loss': torch.mean(outer), 'std_objective_loss': torch.std(outer)}
    
    
    
    def hparam2str(self,hp):
        hpstr = ''
        for key in hp:
            hpstr += ( '\t\t' + key + ' = ' + str(hp[key]) + '\n')
        return hpstr
            

            

    def train(self, train_loader, hp, save_model = False, valid_loader = None):
        
        
        self.set_model(hp, obj_criteria= self.criteria['objective_criteria'])
        
        self.set_optimizer(hp)
        
        if self.logging:
            print('Training started.')
            print('\tModel: '+ 'RapidENet')
            print('\tOptimizer: ' + hp['optimizer'])
            print('\tHyperparameter:\n' + self.hparam2str(hp))


        self.set_train_dict(hp['num_of_epochs'])
        
        
        
        for epoch in range(hp['num_of_epochs']):
            
            if logging:
                print('Epoch '+ str(epoch+1) + ' started.')
            self.model.train()
            # iterating on train batches and update model weights
            #print(train_loader.dataset.df)
            for i, train_batch in enumerate(train_loader):
                #print(len(train_batch))
                #train_batch_target = list(map(lambda x: x[5].to(self.device), train_batch))
                #train_batch_weights = list(map(lambda x: x[6].to(self.device), train_batch))
                # = torch.tensor(list(map(lambda x: x[0][5], train_batch))).to(self.device)

                #print(train_batch)




                scatters = list(map(lambda x: x[0].to(self.device), train_batch))
                spectrums = list(map(lambda x: x[1].to(self.device), train_batch))
                lifetimes1 = list(map(lambda x: x[2].to(self.device), train_batch))
                lifetimes2 = list(map(lambda x: x[3].to(self.device), train_batch))
                sizes = list(map(lambda x: x[4].to(self.device), train_batch))
                target = list(map(lambda x: x[5].to(self.device), train_batch))
                weights = list(map(lambda x: x[6].to(self.device), train_batch))
                #train_batch_weights = torch.tensor(list(map(lambda x: x[2], train_batch))).to(device)

                #objective_batch_loss = self.criteria['objective_criteria'](target, weights)
                #print(objective_batch_loss)
                #print(train_batch_target)
                #train_batch_weights = torch.tensor(list(map(lambda x: x[2], train_batch))).to(self.device)
                
                
                #print('Data:')
                #print(train_batch_target)
                #print(train_batch_weights)


                train_batch_output = self.model(scatters, spectrums, lifetimes1, lifetimes2, sizes, target, weights)
                print(train_batch_output)
                sys.exit();




                #objective_batch_loss = self.criteria['objective_criteria'](train_batch_output, train_batch_target, train_batch_weights)
                #objective_batch_loss.mean().backward()

                #self.optimizer.step(lambda: objective_batch_loss)
                #self.scheduler.step(objective_batch_loss)
                
                #self.update_batch_info('train', train_batch_output, train_batch_target, train_batch_weights, i, epoch, len(train_loader))
                #print("Batch " + str(i+1) +"completed")

            # iterating on valid batches
            self.model.eval()
            if valid_loader:
                for j, valid_batch in enumerate(valid_loader):
                    valid_batch_data, valid_batch_target, valid_batch_weights = valid_batch
                    valid_batch_output = self.model(valid_batch_data)
                    self.update_batch_info('valid', valid_batch_output, valid_batch_target, valid_batch_weights, j, epoch, len(valid_loader))
            
            
            self.update_best_model_for_each_criteria(train_loader.dataset.name, valid_loader.dataset.name if valid_loader != None else None, hp, epoch, save_model)
            
            #print epoch results from statedict
            
            for ds in ['train', 'valid']:
                print("\t\t" + ds + ":")
                for cn in [self.args['objective_criteria']] + self.args['additional_criteria']:
                    print('\t\t\t' + cn + ': ' + str(self.train_dict[ds][cn]['epochs_mean'][epoch]))
            
            
            
            
            
        
        if valid_loader:
            return self.train_dict['valid'][self.args['selection_criteria']]['best_value']
        else:
            return self.train_dict['train'][self.args['selection_criteria']]['best_value']
    
    
    def create_file_name(self, hp, dataloader, criteria_name, dataset_type = 'valid'):
        fname = dataloader.dataset.name +'_' + dataset_type + '_' + criteria_name
        for key in hp:
            fname + '_' + key + '=' + str(hp[key])
        
        return fname + '.pt'
    
    def save_model_state(self, hp, epoch_idx, dataset_name, criteria_name, dataset_type):
        state = copy.deepcopy(hp)
        state['epoch'] = epoch_idx
        state['model_state_dict'] = self.args['model'].state_dict()
        state['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save( state, os.path.join(self.model_dir_path,self.create_file_name(hp,dataset_name,criteria_name, dataset_type)))
    
    def update_best_model_for_each_criteria(self, traindataset_name, valid_dataset_name, hp, epoch_idx, save_model):
        for criteria in [self.criteria['objective_criteria']] + self.criteria['additional_criteria']:
            # red = (criteria.module.reduction if self.args['GPU'] else criteria.reduction)
            # name = (criteria.module.name if self.args['GPU'] else criteria.name)
            red = criteria.reduction
            name = criteria.name
            if (criteria.module.sense if self.args['GPU'] else criteria.sense) == 'min':
                if  self.train_dict['train'][name]['epochs_'+red][epoch_idx] <  self.train_dict['train'][name]['best_value']:
                    self.train_dict['train'][name]['best_value'] =  self.train_dict['train'][name]['epochs_'+red][epoch_idx]
                    if save_model:
                            self.save_model_state(hp, epoch_idx, traindataset_name, 'train')
                if valid_dataset_name:
                    if  self.train_dict['valid'][name]['epochs_'+red][epoch_idx] <  self.train_dict['valid'][name]['best_value']:
                        self.train_dict['valid'][name]['best_value'] =  self.train_dict['valid'][name]['epochs_'+red][epoch_idx]
                        if save_model:
                                self.save_model_state(hp, epoch_idx, valid_dataset_name, 'valid')      
            else:
                if  self.train_dict['train'][name]['epochs_'+red][epoch_idx] >  self.train_dict['train'][name]['best_value']:
                    self.train_dict['train'][name]['best_value'] =  self.train_dict['train'][name]['epochs_'+red][epoch_idx]
                    if save_model:
                            self.save_model_state(hp, epoch_idx, traindataset_name, 'train')
                
                if valid_dataset_name:
                
                    if  self.train_dict['valid'][name]['epochs_'+red][epoch_idx] >  self.train_dict['valid'][name]['best_value']:
                        self.train_dict['valid'][name]['best_value'] =  self.train_dict['valid'][name]['epochs_'+red][epoch_idx]
                        if save_model:
                                self.save_model_state(hp, epoch_idx, valid_dataset_name, 'valid')
                            
    def validate(self, valid_loader, hp,  model_path = None):
        self.set_model(hp, model_path)
        self.model.eval()
        self.set_trainvalid_dict(1)
        for j, valid_batch in enumerate(valid_loader):
                valid_batch_data, valid_batch_target, valid_batch_weights = valid_batch
                valid_batch_output = self.model(valid_batch_data)
                self.update_batch_info('valid', valid_batch_output, valid_batch_target, valid_batch_weights, j, 0, len(valid_loader))
                
    def deploy(self, save_model = True):
        valid_split_groups = np.array(list(self.df['CLUSTER']))
        train_valid = train_test_split(valid_split_groups, num_splits = self.args['num_of_test_splits'])
        inner = torch.zeros(len(self.hyperparameters), self.args['num_of_test_splits'])
        for j, train_data, valid_data in enumerate(train_valid):
            
            df_train = self.df.iloc[sorted(train_data)]
            train_loader = self.prepare_data_loader(df_train, self.args['train_batch_size'], str(j+1) + '_' +'train')
            df_valid = self.df.iloc[sorted(valid_data)]
            valid_loader = self.prepare_data_loader(df_valid, self.args['valid_batch_size'], str(j+1) + '_' +'valid')
            
            for k, hp in enumerate(self.hyperparameters):
                
                inner[k][j] = self.train(train_loader, valid_loader, hp)
                
        hp_best = self.hyperparameters[self.tune_hparam(inner)]
        data_loader = self.prepare_data_loader(self.df, self.args['train_batch_size'], 'deploy')
        result = self.train(data_loader, hp_best, save_model=True)
        return result
        
        
        
        
exp1 = Experiment(args)
exp1.nested_crossvalidation() 

